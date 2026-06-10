"""
Smoke test for Qwen2-VL-72B-AWQ consistency check.

Loads two images and runs LocalVisionChecker.check_character_consistency,
printing the verdict, raw VRAM peak, and per-stage timing.

Usage (from project root):
    python test_qwen72b_consistency.py
    python test_qwen72b_consistency.py <ref_image> <current_image>
    python test_qwen72b_consistency.py <ref_image> <current_image> \\
        --desc "rust-red fox with white belly" --shot "fox in snowy forest"

If no images are passed, the script searches `outputs/` for the first two
keyframes it can find.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Project root must be on path so `src.utils.local_vision` imports.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.local_vision import LocalVisionChecker
try:
    from src.utils.local_vision import DEFAULT_MODEL_PATH
except ImportError:
    # Older local_vision.py (pre-72B-AWQ refactor) did not export this
    # constant. Resolve the same way the new module does so the test
    # works even before the wrapper update is in place.
    import os as _os
    DEFAULT_MODEL_PATH = _os.environ.get(
        "QWEN2VL_MODEL_PATH",
        "/root/autodl-tmp/hf_cache/qwen2vl-72b-awq",
    )


def _vram_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _find_default_keyframes() -> tuple[Path, Path] | None:
    """Look for two PNG keyframes anywhere under outputs/."""
    out_root = ROOT / "outputs"
    if not out_root.exists():
        return None
    pngs = sorted(out_root.rglob("*.png"))
    if len(pngs) < 2:
        return None
    return pngs[0], pngs[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ref",     nargs="?", type=Path, help="Reference image")
    p.add_argument("current", nargs="?", type=Path, help="Current image")
    p.add_argument("--desc", default="a character (use image as description)",
                   help="Character textual description")
    p.add_argument("--shot", default="generic shot for smoke test",
                   help="Shot textual description")
    p.add_argument("--name", default="the character", help="Character name")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                   help="Override Qwen2-VL-72B-AWQ snapshot path")
    p.add_argument("--no-offload", action="store_true",
                   help="Keep model resident after call (default: free VRAM)")
    args = p.parse_args()

    # ---- resolve images ----------------------------------------------------
    if args.ref is None or args.current is None:
        found = _find_default_keyframes()
        if found is None:
            print(
                "ERROR: no images given and outputs/ has fewer than 2 PNG "
                "keyframes. Pass paths explicitly:\n"
                "  python test_qwen72b_consistency.py <ref.png> <current.png>",
                file=sys.stderr,
            )
            return 2
        args.ref, args.current = found
        print(f"[auto] using ref     = {args.ref}")
        print(f"[auto] using current = {args.current}")

    for label, pth in [("ref", args.ref), ("current", args.current)]:
        if not pth.exists():
            print(f"ERROR: {label} image not found: {pth}", file=sys.stderr)
            return 2

    # ---- environment sanity ------------------------------------------------
    print("=" * 60)
    print(f"torch:       {torch.__version__}")
    print(f"CUDA avail:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total VRAM:  {total_gb:.1f} GB")
    print(f"Model path:  {args.model_path}")
    print(f"Ref image:   {args.ref}")
    print(f"Curr image:  {args.current}")
    print(f"Offload:     {not args.no_offload}")
    print("=" * 60)

    if not Path(args.model_path).exists():
        print(f"ERROR: model path does not exist: {args.model_path}",
              file=sys.stderr)
        return 2

    # ---- load & infer ------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    checker = LocalVisionChecker(
        model_id=args.model_path,
        offload_between_calls=not args.no_offload,
    )

    t0 = time.perf_counter()
    checker._load()
    t_load = time.perf_counter() - t0
    print(f"[timing] load:      {t_load:6.2f} s   VRAM peak so far: "
          f"{_vram_peak_gb():.2f} GB")

    t1 = time.perf_counter()
    verdict = checker.check_character_consistency(
        current_image=args.current,
        character_description=args.desc,
        shot_description=args.shot,
        reference_image=args.ref,
        character_name=args.name,
    )
    t_infer = time.perf_counter() - t1
    print(f"[timing] inference: {t_infer:6.2f} s   VRAM peak: "
          f"{_vram_peak_gb():.2f} GB")

    print("=" * 60)
    print("VERDICT:")
    print(f"  consistent          : {verdict.get('consistent')}")
    print(f"  confidence          : {verdict.get('confidence')}")
    issues = verdict.get("issues") or []
    if issues:
        print("  issues:")
        for it in issues:
            print(f"    - {it}")
    hint = verdict.get("refined_prompt_hint")
    if hint:
        print(f"  refined_prompt_hint : {hint}")
    print("=" * 60)
    print(f"[timing] total wall : {t_load + t_infer:6.2f} s")
    print(f"VRAM peak (all)     : {_vram_peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
