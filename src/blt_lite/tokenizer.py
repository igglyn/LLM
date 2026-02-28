from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SpecialTokens:
    bos: str = "<BOS>"
    eos: str = "<EOS>"


class FixedPatchTokenizer:
    """Byte-identity tokenizer with configurable patch length.

    Maps each raw byte value directly to its integer token ID (0..255), plus
    optional BOS/EOS tokens appended at the end of the vocabulary. `patch_size`
    controls how bytes are chunked internally for patch experiments.
    """

    def __init__(self, patch_size: int = 1):
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.patch_size = patch_size
        self.special = SpecialTokens()
        self.byte_vocab_size = 256
        self.special_tokens = [self.special.bos, self.special.eos]

        self.patch_to_id: dict[str, int] = {
            str(i): i for i in range(self.byte_vocab_size)
        }
        self.patch_to_id[self.special.bos] = self.byte_vocab_size
        self.patch_to_id[self.special.eos] = self.byte_vocab_size + 1

        self.id_to_patch: dict[int, str] = {
            idx: tok for tok, idx in self.patch_to_id.items()
        }

    def _iter_patches(self, raw: bytes) -> Iterable[bytes]:
        for i in range(0, len(raw), self.patch_size):
            yield raw[i : i + self.patch_size]

    def fit(self, texts: Iterable[str]) -> None:
        # Identity byte tokenizer has a fixed vocabulary; fit is a no-op.
        _ = texts

    @property
    def bos_id(self) -> int:
        return self.patch_to_id[self.special.bos]

    @property
    def eos_id(self) -> int:
        return self.patch_to_id[self.special.eos]

    @property
    def vocab_len(self) -> int:
        return len(self.patch_to_id)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        raw = text.encode("utf-8")
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for patch in self._iter_patches(raw):
            ids.extend(int(b) for b in patch)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        out = bytearray()
        for idx in ids:
            if idx in (self.bos_id, self.eos_id):
                if not strip_special:
                    marker = self.id_to_patch[idx].encode("utf-8")
                    out.extend(marker)
                continue
            if 0 <= idx < self.byte_vocab_size:
                out.append(idx)
        return bytes(out).decode("utf-8", errors="replace")

    def diagnostics(self, texts: Iterable[str], add_bos: bool = True, add_eos: bool = True) -> dict:
        total = 0
        for text in texts:
            total += len(self.encode(text, add_bos=add_bos, add_eos=add_eos))
        return {
            "tokenizer": "byte_identity",
            "patch_size": self.patch_size,
            "vocab_size": self.vocab_len,
            "byte_vocab_size": self.byte_vocab_size,
            "total_tokens": total,
            "has_unk": False,
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "tokenizer_type": "byte_identity",
            "patch_size": self.patch_size,
            "byte_vocab_size": self.byte_vocab_size,
            "special_tokens": self.special_tokens,
            "patch_to_id": self.patch_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FixedPatchTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tok = cls(patch_size=int(payload.get("patch_size", 1)))
        loaded_vocab = payload.get("patch_to_id")
        if loaded_vocab:
            tok.patch_to_id = {k: int(v) for k, v in loaded_vocab.items()}
            tok.id_to_patch = {v: k for k, v in tok.patch_to_id.items()}
        return tok


ByteIdentityTokenizer = FixedPatchTokenizer
