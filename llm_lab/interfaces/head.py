"""Head interface contract."""

from abc import ABC, abstractmethod

from llm_lab.typing import HiddenTensor, LogitsTensor


class Head(ABC):
    """Interface for projecting hidden states to logits.

    Contract:
    - Input hidden representation ``h`` must have shape ``[B, T, D]`` and dtype
      ``torch.float32`` or ``torch.float16``.
    - Output logits should have shape ``[B, T, V]`` where ``V`` is vocabulary
      size, typically with floating dtype.
    """

    @abstractmethod
    def forward(self, h: HiddenTensor) -> LogitsTensor:
        """Project hidden states to model logits.

        Args:
            h: Hidden tensor of shape ``[B, T, D]`` with floating dtype.

        Returns:
            Logits tensor of shape ``[B, T, V]``.
        """
        raise NotImplementedError
