"""Mixer interface contract."""

from abc import ABC, abstractmethod

from llm_lab.typing import HiddenTensor


class Mixer(ABC):
    """Interface for sequence mixing blocks (Transformer, SSM, etc.).

    Contract:
    - Input hidden representation ``h`` must have shape ``[B, T, D]`` and dtype
      ``torch.float32`` or ``torch.float16``.
    - Output must preserve hidden representation semantics as shape ``[B, T, D]``
      with dtype ``torch.float32`` or ``torch.float16``.
    """

    @abstractmethod
    def forward(self, h: HiddenTensor) -> HiddenTensor:
        """Apply sequence mixing to hidden states.

        Args:
            h: Hidden tensor of shape ``[B, T, D]`` with floating dtype.

        Returns:
            Hidden tensor of shape ``[B, T, D]`` with floating dtype.
        """
        raise NotImplementedError
