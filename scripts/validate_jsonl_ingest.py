"""Quick bring-up script for teacher-JSONL ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _resolve_jsonl_paths(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return []


def _extract_preview_text(
    record: dict[str, object],
    *,
    jsonl_text_field: str | None,
    include_prompt: bool,
) -> str | None:
    if jsonl_text_field:
        value = record.get(jsonl_text_field)
        return None if value is None else str(value)

    structured = record.get("structured_output")
    if isinstance(structured, dict):
        completion = structured.get("completion_text")
        if completion is not None:
            prompt = structured.get("prompt_text")
            if include_prompt and prompt is not None:
                return f"{prompt}\n{completion}"
            return str(completion)

    completion = record.get("completion_text")
    if completion is not None:
        prompt = record.get("prompt_text")
        if include_prompt and prompt is not None:
            return f"{prompt}\n{completion}"
        return str(completion)

    text = record.get("text")
    if text is not None:
        return str(text)

    return None


def _scan_usable_records(
    *,
    jsonl_path: str,
    jsonl_text_field: str | None,
    include_prompt: bool,
) -> tuple[int, int, str | None]:
    files = _resolve_jsonl_paths(jsonl_path)
    usable_records = 0
    total_bytes = 0
    preview: str | None = None

    for file_path in files:
        for line_no, line in enumerate(
            file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {file_path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                continue

            text = _extract_preview_text(
                record,
                jsonl_text_field=jsonl_text_field,
                include_prompt=include_prompt,
            )
            if text is None:
                continue

            usable_records += 1
            total_bytes += len(text.encode("utf-8", errors="replace"))
            if preview is None:
                preview = text

    return usable_records, total_bytes, preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JSONL ingestion without training")
    parser.add_argument("--config", default="configs/examples/tiny_baseline.toml")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--preview-chars", type=int, default=120)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        print(f"missing dependency: {exc}. install project dependencies to run validation.")
        return

    from llm_lab.config.defaults import load_config
    from llm_lab.data.collate import collate_batch
    from llm_lab.data.jsonl_text_dataset import JsonlTextDataset

    cfg = load_config(args.config)
    dataset_type = str(getattr(cfg.data, "dataset_type", "bytes_txt") or "bytes_txt")
    if dataset_type != "jsonl_text":
        raise ValueError(
            "validate_jsonl_ingest.py requires data.dataset_type='jsonl_text' in config"
        )

    jsonl_path = str(getattr(cfg.data, "jsonl_path", "") or "")
    if not jsonl_path:
        raise ValueError("data.jsonl_path is required for dataset_type='jsonl_text'")

    jsonl_text_field = getattr(cfg.data, "jsonl_text_field", None)
    include_prompt = bool(getattr(cfg.data, "include_prompt", False))

    usable_records, total_bytes, preview_text = _scan_usable_records(
        jsonl_path=jsonl_path,
        jsonl_text_field=jsonl_text_field,
        include_prompt=include_prompt,
    )
    if usable_records == 0:
        raise ValueError(
            "no usable text fields found in JSONL. "
            "Expected one of: jsonl_text_field override, structured_output.completion_text, "
            "completion_text, or text."
        )

    dataset = JsonlTextDataset(
        jsonl_path=jsonl_path,
        seq_len=cfg.data.seq_len,
        seed=cfg.train.seed,
        jsonl_text_field=jsonl_text_field,
        include_prompt=include_prompt,
    )
    if len(dataset) == 0:
        raise ValueError(
            "JSONL has usable text records but produced zero chunks. "
            f"total_bytes={total_bytes}, seq_len={cfg.data.seq_len}."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        drop_last=False,
    )

    print(f"usable_records={usable_records} total_bytes={total_bytes} chunks={len(dataset)}")
    if preview_text is not None:
        snippet = preview_text[: args.preview_chars].replace("\n", "\\n")
        print(f"sample decoded preview: {snippet}")

    max_batches = max(1, int(args.batches))
    for idx, batch in enumerate(dataloader):
        if idx >= max_batches:
            break
        print(
            f"batch[{idx}] byte length={int(batch.numel())} "
            f"shape={tuple(batch.shape)} dtype={batch.dtype}"
        )
        first_preview = bytes(batch[0].tolist()).decode("utf-8", errors="replace")
        print(
            "batch[{}] first-sample preview: {}".format(
                idx,
                first_preview[: args.preview_chars].replace("\n", "\\n"),
            )
        )


if __name__ == "__main__":
    main()
