"""
Post Agent (Stage 7).

Final assembly. Uses MoviePy to:
  1. Load each per-shot MP4.
  2. Attach the corresponding audio file (if narration was produced).
  3. Apply transitions between shots (cut / fade_to_black / dissolve).
  4. Write the final MP4 at the target codec/bitrate/resolution.
"""

from __future__ import annotations

from pathlib import Path

from src.agents import BaseAgent
from src.schemas.data_models import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
    Shot,
    TransitionType,
)
from src.utils import get_logger
from src.utils.debug_dump import debug_dir, is_debug_target

log = get_logger("post")


class PostAgent(BaseAgent):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    def process(self, state: PipelineState) -> PipelineState:
        from moviepy.editor import (
            AudioFileClip,
            CompositeAudioClip,
            VideoFileClip,
            concatenate_videoclips,
        )

        shots = [s for s in state.all_shots() if s.video_clip_path]
        if not shots:
            raise RuntimeError("No video clips available for assembly")

        log.info("Assembling %d shots", len(shots))

        clips = []
        for shot in shots:
            clip = VideoFileClip(shot.video_clip_path)
            if shot.audio_clip_path and Path(shot.audio_clip_path).exists():
                audio = AudioFileClip(shot.audio_clip_path)
                # Trim audio to clip length if needed
                audio = audio.subclip(0, min(audio.duration, clip.duration))
                clip = clip.set_audio(audio)
            clip = self._apply_outgoing_transition(clip, shot)
            clips.append(clip)
            # Binary-search debug: dump this shot's POST output as a single clip
            # (transition + resize applied, but no neighbor crossfade). If the
            # full final shows a seam afterimage that this single clip lacks, the
            # defect is a dissolve crossfade overlap, not the shot itself.
            if is_debug_target(state, shot):
                self._dump_single_shot_final(state, shot, clip)

        # Concatenate with crossfade where dissolves were requested
        final = self._concat_with_transitions(clips, shots)

        # Resize to target resolution
        target_w = self.config.video.width
        target_h = self.config.video.height
        if final.w != target_w or final.h != target_h:
            log.info("Resizing %dx%d → %dx%d", final.w, final.h, target_w, target_h)
            final = final.resize(newsize=(target_w, target_h))

        output_path = Path(state.output_dir) / "final_video.mp4"
        log.info("Writing final video: %s", output_path)

        final.write_videofile(
            str(output_path),
            fps=self.config.video.target_fps,
            codec=self.config.video.output_codec,
            audio_codec="aac",
            bitrate=self.config.video.output_bitrate,
            audio_bitrate="192k",
            preset="medium",
            threads=4,
            logger=None,
        )

        # Clean up clip resources
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        try:
            final.close()
        except Exception:
            pass

        state.final_video_path = str(output_path)
        state.advance_to(PipelineStage.EDITING)
        log.info("Final video ready: %s", output_path)
        return state

    # ---- Debug single-shot dump ----------------------------------------

    def _dump_single_shot_final(self, state: PipelineState, shot: Shot, clip) -> None:
        """Write one post-processed shot to <output>/debug/<shotid>_final.mp4.
        Best-effort: any failure is logged but never aborts the main assembly."""
        try:
            tw, th = self.config.video.width, self.config.video.height
            out_clip = clip
            if out_clip.w != tw or out_clip.h != th:
                out_clip = out_clip.resize(newsize=(tw, th))
            out_path = debug_dir(state) / f"{shot.id}_final.mp4"
            out_clip.write_videofile(
                str(out_path),
                fps=self.config.video.target_fps,
                codec=self.config.video.output_codec,
                audio_codec="aac",
                bitrate=self.config.video.output_bitrate,
                preset="medium",
                threads=4,
                logger=None,
            )
            log.info("[debug dump] %s final -> %s", shot.id, out_path)
            if out_clip is not clip:
                out_clip.close()
        except Exception as e:
            log.warning("[debug dump] single-shot final failed for %s: %s", shot.id, e)

    # ---- Transitions ----------------------------------------------------

    def _apply_outgoing_transition(self, clip, shot: Shot):
        ttype = shot.transition_out
        if ttype == TransitionType.FADE_TO_BLACK.value:
            return clip.fadeout(0.4)
        if ttype == TransitionType.FADE_IN.value:
            return clip.fadein(0.4)
        return clip

    def _concat_with_transitions(self, clips: list, shots: list[Shot]):
        from moviepy.editor import concatenate_videoclips

        # Build a list of (clip, method) pairs. For "dissolve", we use compose
        # method on the concatenation, applying a crossfadein on the next clip.
        has_dissolve = any(
            s.transition_out == TransitionType.DISSOLVE.value for s in shots[:-1]
        )
        if not has_dissolve:
            return concatenate_videoclips(clips, method="compose")

        # Apply crossfade where requested
        processed = [clips[0]]
        for i in range(1, len(clips)):
            prev_shot = shots[i - 1]
            curr = clips[i]
            if prev_shot.transition_out == TransitionType.DISSOLVE.value:
                curr = curr.crossfadein(0.5)
            processed.append(curr)
        return concatenate_videoclips(processed, method="compose", padding=-0.5)
