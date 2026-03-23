"""block_float.py — C4F4M2+S B4B9 block float encoder for NNv5 input.

Format: C4F4M2+S B4B9
    C4  — 4-bit coarse exponent (shared across entire 128-bit block)
    F4  — 4-bit fine exponent (shared across each B9 group)
    M2  — 2-bit mantissa per value
    +S  — 1 sign bit per value
    B4  — 4 fine groups per block
    B9  — 9 values per fine group

Layout (128 bits = 2 uint64 words):
    [C4][F4|V9|V9|V9|V9|V9|V9|V9|V9|V9][F4|V9×9][F4|V9×9][F4|V9×9]
    where V = S1M2 (3 bits: sign + 2 mantissa)

    4 + 4×(4 + 9×3) = 4 + 4×31 = 128 bits exactly
    36 values per 128-bit block

Target input range: [-3.0, 3.0] (transformer post-layernorm activations)
"""
from __future__ import annotations
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COARSE_BITS = 4     # bits for coarse exponent
FINE_BITS   = 4     # bits for fine exponent
MANT_BITS   = 2     # bits for mantissa
SIGN_BITS   = 1     # sign bit per value
VALUE_BITS  = SIGN_BITS + MANT_BITS   # 3 bits per value
B_FINE      = 4     # fine groups per block
B_VAL       = 9     # values per fine group
VALUES_PER_BLOCK = B_FINE * B_VAL    # 36 values per 128-bit block
BLOCK_BITS  = COARSE_BITS + B_FINE * (FINE_BITS + B_VAL * VALUE_BITS)  # 128

assert BLOCK_BITS == 128, f"Format error: {BLOCK_BITS} != 128"

# Coarse exponent covers [-3, 3] — biased so 0 maps to exponent -8
# 4 bits → 16 coarse levels, step = 1 bit of exponent
COARSE_BIAS = 8     # coarse_exp = floor(log2(|x|)) + COARSE_BIAS, clamped [0, 15]
FINE_BIAS   = 8     # fine_exp similarly biased within coarse bucket


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_block_float(x: np.ndarray) -> np.ndarray:
    """
    Encode float32 array to C4F4M2+S B4B9 block float format.

    x:       (N, D) float32, values expected in roughly [-3, 3]
    returns: (N, D // VALUES_PER_BLOCK, 2) uint64
             2 uint64 words per 128-bit block

    D must be divisible by VALUES_PER_BLOCK (36).
    Values are clamped to [-3, 3] before encoding.
    """
    N, D = x.shape
    assert D % VALUES_PER_BLOCK == 0, \
        f"D={D} must be divisible by VALUES_PER_BLOCK={VALUES_PER_BLOCK}"
    n_blocks = D // VALUES_PER_BLOCK

    x = np.clip(x, -3.0, 3.0).astype(np.float32)
    out = np.zeros((N, n_blocks, 2), dtype=np.uint64)

    for b in range(n_blocks):
        block = x[:, b * VALUES_PER_BLOCK:(b + 1) * VALUES_PER_BLOCK]  # (N, 36)

        # Coarse exponent — from max abs value in block
        abs_block  = np.abs(block)
        max_abs    = abs_block.max(axis=1, keepdims=True).clip(1e-9)  # (N, 1)
        coarse_exp = np.floor(np.log2(max_abs)).astype(np.int32) + COARSE_BIAS
        coarse_exp = np.clip(coarse_exp, 0, (1 << COARSE_BITS) - 1)   # (N, 1)

        # Scale by coarse exponent
        coarse_scale = np.power(2.0, (coarse_exp - COARSE_BIAS).astype(np.float32))  # (N, 1)
        block_scaled = block / coarse_scale                             # (N, 36)

        # Pack into 128 bits per sample
        for n in range(N):
            bits = np.uint64(0)
            pos  = np.uint64(0)

            # Coarse exponent — 4 bits at position 0
            ce = np.uint64(int(coarse_exp[n, 0]))
            bits |= (ce & np.uint64(0xF)) << pos
            pos  += np.uint64(COARSE_BITS)

            for f in range(B_FINE):
                group = block_scaled[n, f * B_VAL:(f + 1) * B_VAL]  # (9,)

                # Fine exponent — from max abs in group
                max_g = max(abs(group.max()), abs(group.min()), 1e-9)
                fine_exp = int(np.floor(np.log2(max_g))) + FINE_BIAS
                fine_exp = max(0, min(fine_exp, (1 << FINE_BITS) - 1))

                # Fine exponent — 4 bits
                if pos >= np.uint64(64):
                    # Switch to second word
                    word_idx = 1
                    local_pos = pos - np.uint64(64)
                else:
                    word_idx  = 0
                    local_pos = pos

                fe = np.uint64(fine_exp)
                out[n, b, word_idx] |= (fe & np.uint64(0xF)) << local_pos
                pos += np.uint64(FINE_BITS)

                # Scale by fine exponent
                fine_scale = 2.0 ** (fine_exp - FINE_BIAS)
                group_scaled = group / max(fine_scale, 1e-9)

                for v in range(B_VAL):
                    val = group_scaled[v]
                    sign = np.uint64(1) if val < 0 else np.uint64(0)
                    # Quantize abs value to 2-bit mantissa
                    # 2 bits → 4 levels: 0, 0.25, 0.5, 0.75 (normalized to fine scale)
                    mant = np.uint64(min(3, int(abs(val) * 4)))

                    packed = (sign << np.uint64(2)) | mant   # S|M1|M0

                    if pos >= np.uint64(64):
                        word_idx  = 1
                        local_pos = pos - np.uint64(64)
                    else:
                        word_idx  = 0
                        local_pos = pos

                    out[n, b, word_idx] |= (packed & np.uint64(0x7)) << local_pos
                    pos += np.uint64(VALUE_BITS)

    return out  # (N, n_blocks, 2) uint64


def decode_block_float(encoded: np.ndarray, D: int) -> np.ndarray:
    """
    Decode C4F4M2+S B4B9 block float back to float32.

    encoded: (N, n_blocks, 2) uint64
    D:       original embedding dimension
    returns: (N, D) float32 (approximate — lossy quantization)
    """
    N, n_blocks, _ = encoded.shape
    out = np.zeros((N, D), dtype=np.float32)

    for b in range(n_blocks):
        for n in range(N):
            pos = np.uint64(0)

            def read_bits(nbits):
                nonlocal pos
                nb = np.uint64(nbits)
                if pos >= np.uint64(64):
                    word_idx  = 1
                    local_pos = pos - np.uint64(64)
                else:
                    word_idx  = 0
                    local_pos = pos
                mask = (np.uint64(1) << nb) - np.uint64(1)
                val  = (encoded[n, b, word_idx] >> local_pos) & mask
                pos += nb
                return int(val)

            ce         = read_bits(COARSE_BITS)
            coarse_exp = ce - COARSE_BIAS
            coarse_scale = 2.0 ** coarse_exp

            for f in range(B_FINE):
                fe        = read_bits(FINE_BITS)
                fine_exp  = fe - FINE_BIAS
                fine_scale = 2.0 ** fine_exp

                for v in range(B_VAL):
                    packed = read_bits(VALUE_BITS)   # S|M1|M0
                    sign   = -1.0 if (packed >> 2) & 1 else 1.0
                    mant   = packed & 0x3
                    value  = sign * (mant / 4.0) * fine_scale * coarse_scale
                    idx    = b * VALUES_PER_BLOCK + f * B_VAL + v
                    if idx < D:
                        out[n, idx] = value

    return out


def encode_to_uint64_flat(x: np.ndarray) -> np.ndarray:
    """
    Encode and flatten to (N, n_blocks * 2) uint64 for NNv5 input.
    x: (N, D) float32
    """
    encoded = encode_block_float(x)             # (N, n_blocks, 2)
    N, n_blocks, _ = encoded.shape
    return encoded.reshape(N, n_blocks * 2)     # (N, match_words)


def encode_torch(x: torch.Tensor) -> np.ndarray:
    """
    Convenience wrapper — accepts torch tensor, returns uint64 numpy.
    x: (N, D) float32 torch tensor
    """
    return encode_to_uint64_flat(x.cpu().numpy())
