"""Abstract base class for all VLM models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GenerationOutput:
    """Structured output from VLM generation.

    Attributes:
        generated_text: The decoded text output
        tokens: List of token strings (one per generated step)
        token_logprobs: Log-probability of each generated token
        top_logprobs: List of dicts mapping token_string → logprob (top-k at each step)
        full_scores: Optional list of raw logit tensors (vocab_size,) per step
    """
    generated_text: str
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    full_scores: Optional[list] = field(default=None)


class BaseVLM(ABC):
    """Base interface for Vision-Language Models.

    Subclasses wrap specific model families (LLaVA, Qwen-VL, InternVL, etc.)
    and provide a uniform generate() interface.
    """

    @abstractmethod
    def load_model(self, device: str = "cuda:0"):
        """Load the model and processor onto the specified device."""
        pass

    @abstractmethod
    def generate(self, image_path: str, prompt: str) -> GenerationOutput:
        """Run inference on an image+prompt pair.

        Args:
            image_path: Path to the input image
            prompt: Text prompt (question for actor, evaluation prompt for judge)

        Returns:
            GenerationOutput with text, tokens, logprobs, and optionally full scores
        """
        pass

    @abstractmethod
    def unload_model(self):
        """Free GPU memory by deleting model and clearing cache."""
        pass
