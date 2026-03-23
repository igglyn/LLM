"""synthetic_batch — NNv5/NNv2 synthetic batch augmentation."""
from .nnv5 import NNv5Block, NNv5CapacityError
from .nnv2 import NNv2Block
from .pipeline import NNv5LayerConfig, SyntheticBatchHarness

__all__ = [
    "NNv5Block",
    "NNv5CapacityError",
    "NNv2Block",
    "NNv5LayerConfig",
    "SyntheticBatchHarness",
]
