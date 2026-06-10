"""
StoryForge — CLI entry point.

Usage:
    python main.py run "A fox crosses a snowy forest." --duration 20
    python main.py resume <session_id>
    python main.py list-sessions
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from src.pipeline.checkpoint import latest_checkpoint, list_sessions
from src.pipeline.orchestrator import PipelineOrchestrator
from src.schemas.data_models import PipelineConfig

app = typer.Typer(
    add_completion=False,
    help="StoryForge: Multi-Agent Text-to-Video Generation Pipeline",
)


def _load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        rprint(f"[red]Config file not found:[/red] {config_path}")
        raise typer.Exit(code=1)
    return PipelineConfig.from_yaml(config_path)


@app.command()
def run(
    story: str = typer.Argument(..., help="Story text to turn into a video"),
    duration: int = typer.Option(20, help="Target video duration in seconds"),
    style: str = typer.Option(
        "cinematic, photorealistic", help="Art style hint passed to the Director"
    ),
    config: Path = typer.Option(
        Path("config.yaml"), exists=False, help="Path to config.yaml"
    ),
):
    """Generate a video from a story prompt."""
    cfg = _load_config(config)
    orch = PipelineOrchestrator(cfg)
    final = orch.run(story=story, duration=duration, style=style)
    rprint(f"[green]✓ Final video:[/green] {final}")


@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Session id to resume"),
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
):
    """Resume a previously interrupted pipeline run."""
    cfg = _load_config(config)
    orch = PipelineOrchestrator(cfg)
    final = orch.resume(session_id)
    rprint(f"[green]✓ Final video:[/green] {final}")


@app.command("list-sessions")
def list_sessions_cmd(
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
):
    """List all checkpointed sessions."""
    cfg = _load_config(config)
    sessions = list_sessions(cfg.pipeline.checkpoint_dir)
    if not sessions:
        rprint("[yellow]No sessions found.[/yellow]")
        return
    rprint(f"[bold]Sessions in {cfg.pipeline.checkpoint_dir}:[/bold]")
    for sid in sessions:
        info = latest_checkpoint(sid, cfg.pipeline.checkpoint_dir)
        if info:
            stage, _ = info
            rprint(f"  • {sid}  → last stage: [cyan]{stage.value}[/cyan]")
        else:
            rprint(f"  • {sid}  (no checkpoints)")


@app.command("dry-run")
def dry_run(
    story: str = typer.Argument(...),
    duration: int = typer.Option(20),
    style: str = typer.Option("cinematic, photorealistic"),
    config: Path = typer.Option(Path("config.yaml")),
):
    """Run only Stage 1 (Director) to inspect the script breakdown."""
    cfg = _load_config(config)
    orch = PipelineOrchestrator(cfg)
    from src.schemas.data_models import PipelineState

    state = PipelineState.new(
        story=story,
        duration=duration,
        style=style,
        output_root=cfg.pipeline.output_dir,
    )
    director = orch._get_agent("director")  # type: ignore[attr-defined]
    state = director.process(state)
    if state.script_plan is None:
        rprint("[red]Director produced no script.[/red]")
        raise typer.Exit(code=1)
    rprint(state.script_plan.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
