from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<PAD_PATCH>"
    unk: str = "<UNK_PATCH>"
    bos: str = "<BOS>"
    eos: str = "<EOS>"


class FixedPatchTokenizer:
    def __init__(self, patch_size: int = 8, vocab_size: int = 4096, min_freq: int = 1):
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special = SpecialTokens()
        self.special_tokens = [self.special.pad, self.special.unk, self.special.bos, self.special.eos]
        self.patch_to_id: dict[str, int] = {}
        self.id_to_patch: dict[int, str] = {}

    @staticmethod
    def _bytes_to_key(patch: bytes) -> str:
        return patch.hex()

    @staticmethod
    def _key_to_bytes(key: str) -> bytes:
        return bytes.fromhex(key)

    def _iter_patches(self, raw: bytes) -> Iterable[bytes]:
        for i in range(0, len(raw), self.patch_size):
            patch = raw[i : i + self.patch_size]
            if len(patch) < self.patch_size:
                patch = patch + bytes(self.patch_size - len(patch))
            yield patch

    def fit(self, texts: Iterable[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            raw = text.encode("utf-8")
            for patch in self._iter_patches(raw):
                counter[self._bytes_to_key(patch)] += 1

        keep = [
            key
            for key, freq in counter.most_common()
            if freq >= self.min_freq
        ][: max(0, self.vocab_size - len(self.special_tokens))]

        self.patch_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        offset = len(self.patch_to_id)
        for i, key in enumerate(keep):
            self.patch_to_id[key] = offset + i
        self.id_to_patch = {idx: tok for tok, idx in self.patch_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.patch_to_id[self.special.pad]

    @property
    def unk_id(self) -> int:
        return self.patch_to_id[self.special.unk]

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
        if not self.patch_to_id:
            raise RuntimeError("Tokenizer is not fitted.")
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for patch in self._iter_patches(text.encode("utf-8")):
            key = self._bytes_to_key(patch)
            ids.append(self.patch_to_id.get(key, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        chunks: list[bytes] = []
        for idx in ids:
            token = self.id_to_patch.get(idx, self.special.unk)
            if token in self.special_tokens:
                if strip_special:
                    continue
                chunks.append(token.encode("utf-8"))
            else:
                chunks.append(self._key_to_bytes(token))
        raw = b"".join(chunks).rstrip(b"\x00")
        return raw.decode("utf-8", errors="replace")

    def diagnostics(self, texts: Iterable[str], add_bos: bool = True, add_eos: bool = True) -> dict:
        total = 0
        unk = 0
        for text in texts:
            ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            total += len(ids)
            unk += sum(i == self.unk_id for i in ids)
        return {
            "patch_size": self.patch_size,
            "vocab_size": self.vocab_len,
            "total_tokens": total,
            "unk_tokens": unk,
            "unk_rate": float(unk / max(1, total)),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "patch_size": self.patch_size,
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "special_tokens": self.special_tokens,
            "patch_to_id": self.patch_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FixedPatchTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(
            patch_size=payload["patch_size"],
            vocab_size=payload["vocab_size"],
            min_freq=payload["min_freq"],
        )
        tok.patch_to_id = {k: int(v) for k, v in payload["patch_to_id"].items()}
        tok.id_to_patch = {v: k for k, v in tok.patch_to_id.items()}
        return tok
