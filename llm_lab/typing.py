"""Shared tensor typing conventions for llm_lab interfaces.

This module defines canonical aliases and shape/dtype contracts used by modular
sequence model components.

Tensor conventions
------------------
Raw bytes (`BytesTensor`)
    - dtype: ``torch.uint8``
    - shape: ``[B, T]``

Token IDs (`TokenIdsTensor`)
    - dtype: ``torch.int64``
    - shape: ``[B, T]``

Hidden representations (`HiddenTensor`)
    - dtype: ``torch.float32`` or ``torch.float16``
    - shape: ``[B, T, D]``

Patch IDs (`PatchIdsTensor`)
    - dtype: ``torch.int64``
    - shape: ``[B, Tp]``

Patcher input conventions
-------------------------
Raw patcher input (`RawPatcherInputTensor`)
    - dtype: ``torch.uint8``
    - shape: ``[B, T]``

Hidden patcher input (`HiddenPatcherInputTensor`)
    - dtype: floating
    - shape: ``[B, T, D]``

Additional notes
----------------
- ``B`` is batch size.
- ``T`` is sequence length in token/raw-byte space.
- ``Tp`` is patch sequence length.
- ``D`` is hidden dimension.

Notes on runtime dependencies
-----------------------------
To keep interface imports lightweight in fresh environments, this module uses
``typing.Any`` aliases instead of importing ``torch`` at runtime.
"""

from typing import Any

BytesTensor = Any
TokenIdsTensor = Any
HiddenTensor = Any
PatchIdsTensor = Any
LogitsTensor = Any
RawPatcherInputTensor = Any
HiddenPatcherInputTensor = Any
