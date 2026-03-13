from __future__ import annotations

from dataclasses import dataclass

from distill.schemas import TopKPrediction


@dataclass
class DummyLocalBackend:
    def top_k_next_tokens(self, text: str, k: int) -> list[TopKPrediction]:
        tokens = list(reversed(text.split()[-k:] if text.split() else ["<empty>"]))
        while len(tokens) < k:
            tokens.append(f"tok_{len(tokens)}")
        return [TopKPrediction(token=tok, score=1.0 / (idx + 1), rank=idx + 1) for idx, tok in enumerate(tokens[:k])]


class HuggingFaceBackend:
    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("HF backend requires transformers package.") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def top_k_next_tokens(self, text: str, k: int) -> list[TopKPrediction]:
        import torch  # type: ignore

        encoded = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**encoded).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probs, k)
        tokens = self._tokenizer.convert_ids_to_tokens(indices[0].tolist())
        scores = values[0].tolist()
        return [TopKPrediction(token=str(tok), score=float(score), rank=idx + 1) for idx, (tok, score) in enumerate(zip(tokens, scores))]
