"""
Checkpoint save/load for PipelineState.

Each stage transition writes a JSON dump of the full state under
`<checkpoint_dir>/<session_id>/<stage>.json`. On resume, the most recent
checkpoint is loaded and the pipeline continues from the next stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.schemas.data_models import PipelineStage, PipelineState
from src.utils import get_logger

log = get_logger("checkpoint")


def save_checkpoint(
    state: PipelineState,
    stage_name: str,
    checkpoint_root: Path | str = "checkpoints",
) -> Path:
    root = Path(checkpoint_root) / state.session_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stage_name}.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    log.debug("Checkpoint saved: %s", path)
    return path


def load_checkpoint(
    session_id: str,
    stage_name: str,
    checkpoint_root: Path | str = "checkpoints",
) -> PipelineState:
    path = Path(checkpoint_root) / session_id / f"{stage_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    return PipelineState.model_validate_json(path.read_text(encoding="utf-8"))


def latest_checkpoint(
    session_id: str,
    checkpoint_root: Path | str = "checkpoints",
) -> Optional[tuple[PipelineStage, PipelineState]]:
    """Return the most-advanced checkpoint for a session, or None if no
    checkpoints exist for that session."""
    root = Path(checkpoint_root) / session_id
    if not root.exists():
        return None
    order = PipelineStage.ordered()
    for stage in reversed(order):
        path = root / f"{stage.value.lower()}.json"
        if path.exists():
            state = PipelineState.model_validate_json(path.read_text(encoding="utf-8"))
            return stage, state
    return None


def list_sessions(checkpoint_root: Path | str = "checkpoints") -> list[str]:
    root = Path(checkpoint_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
