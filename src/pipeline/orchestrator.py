"""
Pipeline Orchestrator
=====================

Coordinates the 7-stage agent pipeline. Responsibilities:

  1. Initialize a fresh PipelineState or load one from a checkpoint.
  2. Lazy-load each agent class on first use (so partial implementations
     can be tested independently).
  3. Drive the linear pipeline: Director → Cinematographer → Visual →
     Consistency → Animator → Narrator → Post.
  4. Manage the Stage 3↔4 retry loop: a failed consistency check triggers
     prompt refinement (via Cinematographer) and re-generation (via Visual),
     up to `consistency.max_retries_per_shot` times per shot.
  5. Persist state to disk after every completed stage so a crash can be
     resumed without re-running expensive GPU work.
  6. Aggregate timing/quality metrics and emit a final run summary.

The Orchestrator is intentionally agent-agnostic: agents are loaded by
name from `src.agents.*`, and only need to honor the `BaseAgent` protocol.
"""

from __future__ import annotations

import importlib
import logging
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Protocol, runtime_checkable

# UTF-8 ASCII-only formatter for the file handler — avoids the
# "%a日" / Chinese-locale rendering issues seen in some terminals.
_FILE_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

from src.pipeline.checkpoint import (
    latest_checkpoint as _ckpt_latest,
    save_checkpoint as _ckpt_save,
)
from src.schemas.data_models import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
    Shot,
)
from src.utils import setup_logging


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseAgent(Protocol):
    """Every agent class in `src.agents.*` must implement this interface."""

    def process(self, state: PipelineState) -> PipelineState: ...


# Maps the orchestrator's internal agent key to the (module, class) tuple.
# When a new agent module is implemented under `src/agents/`, no orchestrator
# change is needed — just import and the lazy loader picks it up.
AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "director": ("src.agents.director", "DirectorAgent"),
    "cinematographer": ("src.agents.cinematographer", "CinematographerAgent"),
    "visual": ("src.agents.visual", "VisualAgent"),
    "consistency": ("src.agents.consistency", "ConsistencyAgent"),
    "animator": ("src.agents.animator", "AnimatorAgent"),
    "narrator": ("src.agents.narrator", "NarratorAgent"),
    "post": ("src.agents.post", "PostAgent"),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Raised when the pipeline cannot recover from a failure."""


class AgentNotImplementedError(PipelineError):
    """Raised when a stage's agent class has not yet been implemented."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Drives the StoryForge pipeline end-to-end.

    Example:
        config = PipelineConfig.from_yaml("config.yaml")
        orch = PipelineOrchestrator(config)
        final_video = orch.run(
            story="A fox crosses a snowy forest at dusk.",
            duration=20,
            style="cinematic, photorealistic",
        )
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = self._setup_logger(config.pipeline.log_level)
        self._agents: dict[str, BaseAgent] = {}
        self._interrupted = False
        self._state: Optional[PipelineState] = None
        self._install_sigint_handler()

    # ---- Public API -----------------------------------------------------

    def run(self, story: str, duration: int, style: str) -> Path:
        """Run the full pipeline from raw story input to final MP4 path."""
        state = PipelineState.new(
            story=story,
            duration=duration,
            style=style,
            output_root=self.config.pipeline.output_dir,
        )
        self._state = state
        self.logger.info(
            "Starting new pipeline session: %s (target=%ds, style=%r)",
            state.session_id, duration, style,
        )
        return self._run_from_state(state)

    def resume(self, session_id: str) -> Path:
        """Resume an interrupted pipeline from its latest checkpoint."""
        state = self._load_latest_checkpoint(session_id)
        self._state = state
        self.logger.info(
            "Resuming session %s from stage %s",
            session_id, state.stage.value,
        )
        return self._run_from_state(state)

    # ---- Core driver ----------------------------------------------------

    def _attach_session_log_file(self, state: PipelineState) -> None:
        """Attach a per-session FileHandler so the full transcript is
        persisted at `outputs/<session_id>/run.log`. Idempotent — duplicate
        FileHandlers for the same path are skipped."""
        log_path = Path(state.output_dir) / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger("storyforge")
        # Skip if we've already attached a handler to this exact file
        for h in root.handlers:
            if (
                isinstance(h, logging.FileHandler)
                and Path(getattr(h, "baseFilename", "")) == log_path.resolve()
            ):
                return
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FILE_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(fh)
        self.logger.info("Session log → %s", log_path)

    def _run_from_state(self, state: PipelineState) -> Path:
        self._attach_session_log_file(state)
        try:
            # Each stage method is idempotent w.r.t. its starting stage:
            # if state is already past that stage, it's a no-op.
            self._run_stage_scripting(state)
            self._run_stage_planning(state)
            self._run_stage_generating_with_checking(state)
            self._run_stage_interpolating(state)
            self._run_stage_narrating(state)
            self._run_stage_editing(state)

            self._print_summary(state)
            if state.final_video_path is None:
                raise PipelineError("Pipeline finished but no final video was produced")
            return Path(state.final_video_path)

        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user. Saving checkpoint…")
            self._save_checkpoint(state, "interrupted")
            raise
        except Exception as e:
            self.logger.error("Pipeline failed: %s", e)
            self.logger.debug("%s", traceback.format_exc())
            state.mark_failed(str(e))
            self._save_checkpoint(state, "failed")
            raise

    # ---- Stage implementations -----------------------------------------

    def _run_stage_scripting(self, state: PipelineState) -> None:
        """Stage 1: raw story → ScriptPlan."""
        if self._stage_already_done(state, PipelineStage.SCRIPTING):
            return
        with self._timed_stage(state, PipelineStage.SCRIPTING):
            agent = self._get_agent("director")
            new_state = agent.process(state)
            self._copy_state(new_state, state)
        state.total_shots = len(state.all_shots())
        self._save_checkpoint(state, PipelineStage.SCRIPTING.value)

    def _run_stage_planning(self, state: PipelineState) -> None:
        """Stage 2: ScriptPlan → shots with FLUX prompts."""
        if self._stage_already_done(state, PipelineStage.PLANNING):
            return
        with self._timed_stage(state, PipelineStage.PLANNING):
            agent = self._get_agent("cinematographer")
            new_state = agent.process(state)
            self._copy_state(new_state, state)
        self._save_checkpoint(state, PipelineStage.PLANNING.value)

    def _run_stage_generating_with_checking(self, state: PipelineState) -> None:
        """Stages 3+4: keyframe generation interleaved with consistency
        checking, including the per-shot retry loop."""
        if self._stage_already_done(state, PipelineStage.CHECKING):
            return

        state.advance_to(PipelineStage.GENERATING)
        visual = self._get_agent("visual")
        consistency = self._get_agent("consistency")
        cinematographer = self._get_agent("cinematographer")

        max_retries = self.config.consistency.max_retries_per_shot
        stage_start = time.time()

        if getattr(self.config.consistency, "batch_mode", False):
            self._batch_generate_and_check(
                state, visual, consistency, cinematographer, max_retries
            )
        else:
            for shot in state.all_shots():
                if self._interrupted:
                    raise KeyboardInterrupt
                if shot.approved:
                    # Already done in an earlier resumed run.
                    continue
                self._generate_and_check_shot(
                    shot, state, visual, consistency, cinematographer, max_retries
                )

        # Count approved shots and roll up totals.
        state.approved_shots = sum(1 for s in state.all_shots() if s.approved)
        state.total_retries = sum(s.retry_count for s in state.all_shots())
        state.stage_durations_sec[PipelineStage.GENERATING.value] = time.time() - stage_start
        state.advance_to(PipelineStage.CHECKING)
        self._save_checkpoint(state, PipelineStage.CHECKING.value)

    def _generate_and_check_shot(
        self,
        shot: Shot,
        state: PipelineState,
        visual: BaseAgent,
        consistency: BaseAgent,
        cinematographer: BaseAgent,
        max_retries: int,
    ) -> None:
        """Inner retry loop for a single shot."""
        while not shot.approved and shot.retry_count <= max_retries:
            self.logger.info(
                "[%s] generating (attempt %d/%d)",
                shot.id, shot.retry_count + 1, max_retries + 1,
            )
            # Visual agent generates keyframes for THIS shot only.
            # By convention `process()` looks at `state.stage` and only
            # operates on un-generated shots; agents may also expose a
            # `generate_shot(state, shot)` shortcut, which we prefer.
            self._run_visual_for_shot(visual, state, shot)

            self.logger.info("[%s] checking consistency", shot.id)
            self._run_consistency_for_shot(consistency, state, shot)

            if shot.approved:
                self.logger.info(
                    "[%s] approved (score=%.3f)",
                    shot.id, shot.consistency_score,
                )
                return

            if shot.retry_count >= max_retries:
                self.logger.warning(
                    "[%s] max retries reached (score=%.3f); force-approving",
                    shot.id, shot.consistency_score,
                )
                shot.approved = True
                return

            # Trigger prompt refinement and try again.
            feedback = shot.failure_reasons[-1] if shot.failure_reasons else "low score"
            self.logger.info("[%s] refining prompt: %s", shot.id, feedback)
            self._refine_prompt_for_shot(cinematographer, state, shot, feedback)
            shot.retry_count += 1

    def _batch_generate_and_check(
        self,
        state: PipelineState,
        visual: BaseAgent,
        consistency: BaseAgent,
        cinematographer: BaseAgent,
        max_retries: int,
    ) -> None:
        """
        Batch topology for Stages 3+4 (preferred when Level-2 backend is
        local Qwen-72B-AWQ — amortizes the 17 s VLM reload across all shots).

        Round loop:
          for r in 0..max_retries:
              pending = shots not approved
              if empty: break
              visual.generate_shot for each pending     # FLUX resident
              consistency.evaluate_shots_batch(pending) # VLM resident once
              for each still-failing: refine prompt + bump retry_count
          force-approve any shot still failing at end of loop.
        """
        all_shots = state.all_shots()
        batch_method = getattr(consistency, "evaluate_shots_batch", None)
        if not callable(batch_method):
            # Fall back to per-shot mode if the agent doesn't implement batch.
            self.logger.warning(
                "consistency agent lacks evaluate_shots_batch; falling back to per-shot"
            )
            for shot in all_shots:
                if shot.approved:
                    continue
                self._generate_and_check_shot(
                    shot, state, visual, consistency, cinematographer, max_retries
                )
            return

        for round_idx in range(max_retries + 1):
            if self._interrupted:
                raise KeyboardInterrupt
            pending = [s for s in all_shots if not s.approved]
            if not pending:
                break

            self.logger.info(
                "[batch round %d/%d] generating %d shot(s)",
                round_idx + 1, max_retries + 1, len(pending),
            )
            for shot in pending:
                self._run_visual_for_shot(visual, state, shot)

            self.logger.info(
                "[batch round %d/%d] consistency check on %d shot(s)",
                round_idx + 1, max_retries + 1, len(pending),
            )
            results = batch_method(state, pending)
            for shot, (passed, feedback) in zip(pending, results):
                if passed:
                    shot.approved = True
                    self.logger.info(
                        "[%s] approved (score=%.3f)",
                        shot.id, shot.consistency_score,
                    )
                else:
                    shot.failure_reasons.append(feedback)

            # Refine prompts for shots still failing — unless this is the
            # last allowed round (force-approve handled below).
            if round_idx < max_retries:
                for shot in pending:
                    if shot.approved:
                        continue
                    feedback = shot.failure_reasons[-1] if shot.failure_reasons else "low score"
                    self.logger.info(
                        "[%s] refining prompt: %s", shot.id, feedback,
                    )
                    self._refine_prompt_for_shot(
                        cinematographer, state, shot, feedback
                    )
                    shot.retry_count += 1

        # Force-approve any remaining stragglers.
        for shot in all_shots:
            if not shot.approved:
                self.logger.warning(
                    "[%s] max retries reached (score=%.3f); force-approving",
                    shot.id, shot.consistency_score,
                )
                shot.approved = True

    def _run_stage_interpolating(self, state: PipelineState) -> None:
        """Stage 5: keyframes → per-shot video clips (RIFE)."""
        if self._stage_already_done(state, PipelineStage.INTERPOLATING):
            return
        with self._timed_stage(state, PipelineStage.INTERPOLATING):
            agent = self._get_agent("animator")
            new_state = agent.process(state)
            self._copy_state(new_state, state)
        self._save_checkpoint(state, PipelineStage.INTERPOLATING.value)

    def _run_stage_narrating(self, state: PipelineState) -> None:
        """Stage 6: TTS audio per shot. Optional based on config."""
        if self._stage_already_done(state, PipelineStage.NARRATING):
            return
        if not self.config.tts.enabled:
            self.logger.info("TTS disabled; skipping narration stage")
            state.advance_to(PipelineStage.NARRATING)
            self._save_checkpoint(state, PipelineStage.NARRATING.value)
            return
        with self._timed_stage(state, PipelineStage.NARRATING):
            agent = self._get_agent("narrator")
            new_state = agent.process(state)
            self._copy_state(new_state, state)
        self._save_checkpoint(state, PipelineStage.NARRATING.value)

    def _run_stage_editing(self, state: PipelineState) -> None:
        """Stage 7: assemble clips + audio into final MP4."""
        if self._stage_already_done(state, PipelineStage.EDITING):
            return
        with self._timed_stage(state, PipelineStage.EDITING):
            agent = self._get_agent("post")
            new_state = agent.process(state)
            self._copy_state(new_state, state)
        state.advance_to(PipelineStage.DONE)
        self._save_checkpoint(state, PipelineStage.DONE.value)

    # ---- Agent adapters -------------------------------------------------
    #
    # Visual/Consistency/Cinematographer agents need per-shot calls in the
    # retry loop. If an agent exposes a `*_shot()` method, prefer it;
    # otherwise fall back to the standard `process(state)` interface.

    def _run_visual_for_shot(
        self, agent: BaseAgent, state: PipelineState, shot: Shot
    ) -> None:
        method = getattr(agent, "generate_shot", None)
        if callable(method):
            method(state, shot)
        else:
            agent.process(state)

    def _run_consistency_for_shot(
        self, agent: BaseAgent, state: PipelineState, shot: Shot
    ) -> None:
        method = getattr(agent, "evaluate_shot", None)
        if callable(method):
            passed, feedback = method(state, shot)
            if passed:
                shot.approved = True
            else:
                shot.failure_reasons.append(feedback)
        else:
            agent.process(state)

    def _refine_prompt_for_shot(
        self, agent: BaseAgent, state: PipelineState, shot: Shot, feedback: str
    ) -> None:
        method = getattr(agent, "refine_prompt", None)
        if callable(method):
            new_prompt = method(shot, feedback, state)
            shot.flux_prompt = new_prompt
        else:
            self.logger.warning(
                "Cinematographer has no refine_prompt method; "
                "retry will use the same prompt"
            )

    # ---- Lazy agent loading --------------------------------------------

    def _get_agent(self, key: str) -> BaseAgent:
        if key in self._agents:
            return self._agents[key]
        if key not in AGENT_REGISTRY:
            raise PipelineError(f"Unknown agent key: {key!r}")

        module_name, class_name = AGENT_REGISTRY[key]
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise AgentNotImplementedError(
                f"Agent module {module_name!r} not yet implemented "
                f"({e}). Implement it under src/agents/ and ensure it "
                f"exports the class {class_name!r}."
            ) from e

        try:
            agent_cls = getattr(module, class_name)
        except AttributeError as e:
            raise AgentNotImplementedError(
                f"Module {module_name!r} does not export {class_name!r}"
            ) from e

        agent = agent_cls(self.config)
        if not isinstance(agent, BaseAgent):
            self.logger.warning(
                "Agent %s does not satisfy BaseAgent protocol; calls may fail",
                class_name,
            )
        self._agents[key] = agent
        self.logger.debug("Loaded agent: %s.%s", module_name, class_name)
        return agent

    # ---- Checkpointing --------------------------------------------------

    def _checkpoint_path(self, session_id: str, stage_name: str) -> Path:
        return (
            Path(self.config.pipeline.checkpoint_dir)
            / session_id
            / f"{stage_name}.json"
        )

    def _save_checkpoint(self, state: PipelineState, stage_name: str) -> Path:
        path = _ckpt_save(state, stage_name, self.config.pipeline.checkpoint_dir)
        self.logger.debug("Saved checkpoint: %s", path)
        return path

    def _load_latest_checkpoint(self, session_id: str) -> PipelineState:
        info = _ckpt_latest(session_id, self.config.pipeline.checkpoint_dir)
        if info is None:
            raise PipelineError(f"No checkpoints found for session {session_id!r}")
        stage, state = info
        self.logger.info("Loaded checkpoint at stage %s", stage.value)
        return state

    # ---- Helpers --------------------------------------------------------

    @staticmethod
    def _stage_already_done(state: PipelineState, target: PipelineStage) -> bool:
        """True iff `state.stage` is already past or equal to `target`."""
        order = PipelineStage.ordered()
        if state.stage not in order or target not in order:
            return False
        return order.index(state.stage) > order.index(target)

    @staticmethod
    def _copy_state(src: PipelineState, dst: PipelineState) -> None:
        """Copy all mutable fields from `src` into `dst` in place. Agents
        may return either the same instance or a new one — this normalizes
        the contract so the orchestrator can keep one canonical state."""
        if src is dst:
            return
        for field_name in src.model_fields:
            setattr(dst, field_name, getattr(src, field_name))

    @contextmanager
    def _timed_stage(
        self, state: PipelineState, stage: PipelineStage
    ) -> Iterator[None]:
        """Advance the stage, time the block, and record the duration."""
        state.advance_to(stage)
        self.logger.info("=== Stage: %s ===", stage.value)
        t0 = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - t0
            state.stage_durations_sec[stage.value] = elapsed
            self.logger.info("=== %s done in %.1fs ===", stage.value, elapsed)

    # ---- Summary --------------------------------------------------------

    def _print_summary(self, state: PipelineState) -> None:
        total = sum(state.stage_durations_sec.values())
        approved = state.approved_shots
        total_shots = state.total_shots
        retries = state.total_retries

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE — session %s", state.session_id)
        self.logger.info("=" * 60)
        self.logger.info("Final video : %s", state.final_video_path)
        self.logger.info("Shots       : %d approved / %d total", approved, total_shots)
        self.logger.info("Retries     : %d total across all shots", retries)
        self.logger.info("Wall time   : %.1f s (%.1f min)", total, total / 60)
        self.logger.info("Per-stage timings:")
        for stage_name, dur in state.stage_durations_sec.items():
            self.logger.info("  %-15s %7.1f s", stage_name, dur)
        self.logger.info("=" * 60)

    # ---- Signal handling ------------------------------------------------

    def _install_sigint_handler(self) -> None:
        """Catch Ctrl+C; the next safe point in the retry loop will exit."""

        def handler(signum, frame):
            if self._interrupted:
                # Second Ctrl+C → exit immediately.
                self.logger.error("Second interrupt received; exiting")
                sys.exit(130)
            self._interrupted = True
            self.logger.warning(
                "Interrupt received. Finishing current shot, then saving "
                "checkpoint and exiting. Press Ctrl+C again to force quit."
            )

        try:
            signal.signal(signal.SIGINT, handler)
        except ValueError:
            # signal.signal only works in main thread; ignore in tests
            pass

    @staticmethod
    def _setup_logger(level: str) -> logging.Logger:
        return setup_logging(level)
