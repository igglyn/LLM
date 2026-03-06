"""Memory interface contract."""

from abc import ABC, abstractmethod

from llm_lab.typing import HiddenTensor


class Memory(ABC):
    """Interface for optional external/long-context memory.

    Implementations can store and retrieve hidden representations for extended
    context beyond the current sequence window.

    Contract:
    - ``read`` returns hidden tensor with shape ``[B, Tm, D]`` and floating
      dtype, where ``Tm`` is implementation-defined memory length.
    - ``write`` accepts hidden tensor with shape ``[B, T, D]`` and floating
      dtype for persistence.
    """

    @abstractmethod
    def read(self) -> HiddenTensor:
        """Read memory state.

        Returns:
            Hidden tensor of shape ``[B, Tm, D]`` with floating dtype.
        """
        raise NotImplementedError

    @abstractmethod
    def write(self, h: HiddenTensor) -> None:
        """Write hidden states into memory.

        Args:
            h: Hidden tensor of shape ``[B, T, D]`` with floating dtype.
        """
        raise NotImplementedError
