"""Agent implementations."""

from abc import ABC, abstractmethod

from src.schemas.data_models import PipelineConfig, PipelineState


class BaseAgent(ABC):
    """Common base class for all StoryForge agents."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    @abstractmethod
    def process(self, state: PipelineState) -> PipelineState:
        """Read what is needed from `state`, perform work, write results back."""


__all__ = ["BaseAgent"]
