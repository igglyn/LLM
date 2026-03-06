"""Codec interface contract."""

from abc import ABC, abstractmethod

from llm_lab.typing import BytesTensor, TokenIdsTensor


class Codec(ABC):
    """Interface for raw-byte <-> token-id conversion.

    Contract:
    - ``encode`` input must be raw bytes tensor ``x_u8`` with shape ``[B, T]``
      and dtype ``torch.uint8``.
    - ``encode`` output must be token IDs with shape ``[B, T]`` and dtype
      ``torch.int64``.
    - ``decode`` input must be token IDs with shape ``[B, T]`` and dtype
      ``torch.int64``.
    - ``decode`` output must be raw bytes tensor with shape ``[B, T]`` and
      dtype ``torch.uint8``.
    """

    @abstractmethod
    def encode(self, x_u8: BytesTensor) -> TokenIdsTensor:
        """Encode raw bytes to token IDs.

        Args:
            x_u8: Raw byte tensor of shape ``[B, T]`` and dtype ``torch.uint8``.

        Returns:
            Token ID tensor of shape ``[B, T]`` and dtype ``torch.int64``.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: TokenIdsTensor) -> BytesTensor:
        """Decode token IDs back to raw bytes.

        Args:
            token_ids: Token ID tensor of shape ``[B, T]`` and dtype
                ``torch.int64``.

        Returns:
            Raw byte tensor of shape ``[B, T]`` and dtype ``torch.uint8``.
        """
        raise NotImplementedError

    def quantize_hidden(self, h):
        """Optional hidden-state quantization hook.

        Default behavior is identity so non-quantizing codecs remain unchanged.
        Input/output shape convention: ``[B, T, D]`` floating tensor.
        """
        return h

    def quantize_hidden_with_codes(self, h):
        """Optional quantization hook returning discrete code indices.

        Default behavior uses ``quantize_hidden`` and returns no codes.
        Returns:
            tuple(quantized_hidden, code_indices_or_none)
        """
        return self.quantize_hidden(h), None
