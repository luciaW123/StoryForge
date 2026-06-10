"""
Debug artifact dumper — binary-search localization of per-shot visual bugs.

Enable by exporting DEBUG_DUMP_SHOT, then run the pipeline. For the targeted
shot, three artifacts are copied to <output_dir>/debug/ with fixed names so
you can open them side by side:

    <shotid>_flux_keyframe.png    Visual stage output  (BEFORE Animator)
    <shotid>_cogvideo_raw.mp4     Animator output      (BEFORE Post)
    <shotid>_final.mp4            Post output          (single shot, AFTER transitions)

Find the EARLIEST stage where the defect appears:
    keyframe dirty .................. FLUX / prompt
        → scan the shot's positive prompt for style words like
          chromatic aberration / RGB split / glitch / datamosh / VHS /
          anamorphic / lens distortion / dreamy; check for a stray style LoRA;
          confirm the negative prompt isn't empty.
          Fix: drop the style words, add "chromatic aberration, rgb split,
          color fringing" to the negative prompt.
    keyframe clean, raw dirty ....... CogVideoX temporal instability
        → side-profile / high-frequency edges smear on extrapolation.
          Lower guidance_scale / motion CFG; sanity-check num_frames & fps.
    raw clean, final dirty .......... Post filter graph
        → MoviePy here only does fade/crossfade/concat/resize (no RGB-splitting
          filter). If the defect shows ONLY at a shot seam (3→4 or 4→end) it's a
          dissolve crossfade overlap — change that shot's transition_out.

Target selection (DEBUG_DUMP_SHOT):
    "4"            → 4th shot in global execution order (1-based)
    "scene_0_sh3"  → exact shot id
    "all"          → dump every shot
    unset / ""     → disabled (no-op, zero overhead)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from src.utils import get_logger

log = get_logger("debug_dump")


def _target() -> str:
    return (os.environ.get("DEBUG_DUMP_SHOT", "") or "").strip()


def is_debug_target(state, shot) -> bool:
    """True if `shot` matches DEBUG_DUMP_SHOT (by 1-based index, exact id, or 'all')."""
    t = _target()
    if not t:
        return False
    if t == "all" or t == shot.id:
        return True
    if t.isdigit():
        shots = state.all_shots()
        idx = int(t) - 1  # 1-based, matches human "Shot 4"
        if 0 <= idx < len(shots):
            return shots[idx].id == shot.id
    return False


def debug_dir(state) -> Path:
    d = Path(state.output_dir) / "debug"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump_artifact(state, shot, kind: str, src_path) -> Optional[Path]:
    """
    If `shot` is the debug target, copy `src_path` → <output>/debug/<shotid>_<kind>.
    `kind` is the fixed suffix, e.g. "flux_keyframe.png" or "cogvideo_raw.mp4".
    No-op (returns None) when debugging is disabled or the shot isn't targeted.
    """
    if not is_debug_target(state, shot):
        return None
    src = Path(src_path)
    if not src.exists():
        log.warning("debug dump: source missing for %s %s: %s", shot.id, kind, src)
        return None
    dst = debug_dir(state) / f"{shot.id}_{kind}"
    shutil.copyfile(src, dst)
    log.info("[debug dump] %s %s -> %s", shot.id, kind, dst)
    return dst
