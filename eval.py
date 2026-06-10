"""
Evaluation script — computes the report's quantitative metrics for a session.

Usage:
    python eval.py outputs/<session_id>
    python eval.py outputs/<session_id> --out results/<session_id>.json

Metrics:
    CLIP-T : text-image similarity averaged over sampled frames vs shot prompts
    CLIP-I : cross-shot character similarity vs reference embeddings
    MOTION : mean optical-flow magnitude across the final video
    TIME   : wall-clock generation time (from checkpoints)
    RETRY  : average consistency-retry count per shot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))


def load_state(session_dir: Path) -> dict:
    """Load the latest checkpoint state for this session."""
    chk_root = Path("checkpoints") / session_dir.name
    if not chk_root.exists():
        # Allow custom layouts
        cand = list((session_dir.parent.parent / "checkpoints" / session_dir.name).glob("*.json"))
    else:
        cand = list(chk_root.glob("*.json"))
    if not cand:
        raise FileNotFoundError(f"No checkpoints for session {session_dir.name}")
    # Pick the latest by mtime
    latest = max(cand, key=lambda p: p.stat().st_mtime)
    return json.loads(latest.read_text())


def sample_frames(video_path: Path, n_samples: int = 8) -> list[np.ndarray]:
    """Sample evenly-spaced frames from a video (BGR)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def compute_motion(video_path: Path, max_frames: int = 60) -> float:
    """Mean optical-flow magnitude across consecutive frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    prev_gray = None
    mags: list[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                    poly_n=5, poly_sigma=1.2, flags=0
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
                mags.append(float(mag))
            prev_gray = gray
        idx += 1
    cap.release()
    return float(np.mean(mags)) if mags else 0.0


def compute_clip_metrics(state: dict, session_dir: Path) -> dict:
    """
    Compute CLIP-T (text-image similarity vs shot prompt)
    and CLIP-I (cross-shot character similarity vs reference embedding).
    """
    from src.models.clip_wrapper import get_clip
    clip = get_clip()

    script = state.get("script_plan") or {}
    char_refs: dict[str, np.ndarray] = {}
    for char in script.get("characters", []):
        emb = char.get("clip_embedding")
        if emb:
            char_refs[char["id"]] = np.array(emb, dtype=np.float32)

    clip_t_per_shot: list[float] = []
    clip_i_per_shot: list[float] = []

    shots = []
    for scene in script.get("scenes", []):
        shots.extend(scene.get("shots", []))

    for shot in shots:
        clip_path = shot.get("video_clip_path")
        prompt = shot.get("flux_prompt") or shot.get("description") or ""
        if not clip_path or not Path(clip_path).exists():
            continue

        frames = sample_frames(Path(clip_path), n_samples=8)
        if not frames:
            continue

        # CLIP-T: sample frames vs prompt
        try:
            text_emb = clip.encode_text(prompt) if hasattr(clip, "encode_text") else None
            if text_emb is not None:
                sims = []
                for f in frames:
                    # cv2 returns BGR; convert to PIL RGB for CLIP
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    from PIL import Image
                    pil = Image.fromarray(rgb)
                    img_emb = clip.encode_image(pil) if hasattr(clip, "encode_image") else None
                    if img_emb is not None:
                        sims.append(clip.cosine_similarity(img_emb, text_emb))
                if sims:
                    clip_t_per_shot.append(float(np.mean(sims)))
        except Exception as e:
            print(f"  CLIP-T failed for shot {shot.get('id')}: {e}")

        # CLIP-I: mean shot embedding vs each character's reference
        try:
            embs = []
            for f in frames:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                from PIL import Image
                img_emb = clip.encode_image(Image.fromarray(rgb))
                embs.append(img_emb)
            shot_emb = np.mean(np.stack(embs), axis=0)
            shot_emb = shot_emb / max(float(np.linalg.norm(shot_emb)), 1e-8)

            sims = []
            for cid in shot.get("characters_present", []):
                ref = char_refs.get(cid)
                if ref is not None:
                    sims.append(clip.cosine_similarity(shot_emb, ref))
            if sims:
                clip_i_per_shot.append(float(np.mean(sims)))
        except Exception as e:
            print(f"  CLIP-I failed for shot {shot.get('id')}: {e}")

    return {
        "clip_t_mean": float(np.mean(clip_t_per_shot)) if clip_t_per_shot else None,
        "clip_t_per_shot": clip_t_per_shot,
        "clip_i_mean": float(np.mean(clip_i_per_shot)) if clip_i_per_shot else None,
        "clip_i_per_shot": clip_i_per_shot,
    }


def compute_retry_stats(state: dict) -> dict:
    shots = []
    for scene in state.get("script_plan", {}).get("scenes", []):
        shots.extend(scene.get("shots", []))
    retries = [s.get("retry_count", 0) for s in shots]
    return {
        "retry_mean": float(np.mean(retries)) if retries else 0.0,
        "retry_total": int(sum(retries)),
        "n_shots": len(shots),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path,
                        help="Path to outputs/<session_id>")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write JSON metrics here")
    parser.add_argument("--no-clip", action="store_true",
                        help="Skip CLIP metrics (faster; e.g., for MOTION-only)")
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    if not session_dir.exists():
        print(f"Session dir not found: {session_dir}")
        return 1

    print(f"Evaluating: {session_dir}")

    state = load_state(session_dir)
    print(f"  session_id = {state.get('session_id')}")
    print(f"  stage      = {state.get('stage')}")

    metrics: dict[str, Any] = {
        "session_id": state.get("session_id"),
        "session_dir": str(session_dir),
    }

    # Time
    stage_durations = state.get("stage_durations", {})
    total_time = sum(stage_durations.values()) if stage_durations else None
    metrics["stage_durations_sec"] = stage_durations
    metrics["total_time_sec"] = total_time
    if total_time:
        print(f"  total time = {total_time/60:.1f} min")

    # Retries
    metrics.update(compute_retry_stats(state))
    print(f"  shots = {metrics['n_shots']}, retries = {metrics['retry_total']}"
          f" (avg {metrics['retry_mean']:.2f})")

    # CLIP metrics (slow — loads CLIP model)
    if not args.no_clip:
        print("Computing CLIP metrics...")
        clip_metrics = compute_clip_metrics(state, session_dir)
        metrics.update(clip_metrics)
        print(f"  CLIP-T = {clip_metrics.get('clip_t_mean')}")
        print(f"  CLIP-I = {clip_metrics.get('clip_i_mean')}")

    # MOTION on the final video
    final = session_dir / "final_video.mp4"
    if final.exists():
        print("Computing MOTION (optical flow)...")
        motion = compute_motion(final)
        metrics["motion"] = motion
        print(f"  MOTION = {motion:.3f}")
    else:
        print(f"  final_video.mp4 not found at {final}")
        metrics["motion"] = None

    print("\n=== Summary ===")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("clip_t_per_shot", "clip_i_per_shot",
                                   "stage_durations_sec")},
                     indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nWrote metrics to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
