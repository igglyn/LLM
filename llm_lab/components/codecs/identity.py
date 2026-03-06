"""Identity codec implementation."""

from __future__ import annotations

import torch

from llm_lab.interfaces.codec import Codec
from llm_lab.registry import register_codec


@register_codec("identity")
class IdentityCodec(Codec):
    """Trivial byte<->token codec.

    - ``encode`` casts ``torch.uint8`` bytes to ``torch.int64`` token ids.
    - ``decode`` casts ``torch.int64`` token ids back to ``torch.uint8`` bytes.
    """

    def encode(self, x_u8: torch.Tensor) -> torch.Tensor:
        if x_u8.dtype != torch.uint8:
            raise ValueError("IdentityCodec.encode expects torch.uint8 input.")
        return x_u8.to(torch.int64)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.int64:
            raise ValueError("IdentityCodec.decode expects torch.int64 input.")
        return token_ids.to(torch.uint8)
