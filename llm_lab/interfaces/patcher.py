"""Patcher interface contracts."""

from abc import ABC, abstractmethod

from llm_lab.typing import HiddenPatcherInputTensor, HiddenTensor, RawPatcherInputTensor


class RawPatcher(ABC):
    """Interface for first-stage patchers that consume raw bytes.

    Contract:
    - Input must be raw bytes ``x_u8`` with shape ``[B, T]`` and dtype
      ``torch.uint8``.
    - Output must be hidden states with shape ``[B, Tp, D]`` and floating dtype.
    - ``Tp`` may be fixed or dynamic depending on patch boundary strategy
      (e.g., heuristic/learned boundary patchers).
    """

    @abstractmethod
    def forward(self, x_u8: RawPatcherInputTensor) -> HiddenTensor:
        """Convert raw bytes into hidden patch representations."""
        raise NotImplementedError


class HiddenPatcher(ABC):
    """Interface for second-stage patchers that consume hidden states.

    Contract:
    - Input must be hidden states ``h`` with shape ``[B, T, D]`` and floating
      dtype.
    - Output must be hidden states with shape ``[B, Tp, D]`` and floating dtype.
    - ``Tp`` may be fixed or dynamic depending on patch boundary strategy
      (e.g., heuristic/learned boundary patchers).
    """

    @abstractmethod
    def forward(self, h: HiddenPatcherInputTensor) -> HiddenTensor:
        """Transform hidden patch sequence into a new hidden patch sequence."""
        raise NotImplementedError


class Patcher(RawPatcher):
    """Legacy alias for raw-byte patchers.

    Existing patchers that import/use ``Patcher`` continue to function as
    first-stage raw patchers, including dynamic-boundary implementations.
    """

    pass
