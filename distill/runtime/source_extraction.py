from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

from shared.config.specs import ConfigSpec, DatasetEntrySpec, FilterSpec

from .types import ExtractedDocument


class SourceExtractionError(ValueError):
    pass


def run_source_extraction(config: ConfigSpec) -> list[ExtractedDocument]:
    docs: list[ExtractedDocument] = []
    for dataset_entry in config.dataset.source_extraction.dataset_entries:
        max_entries = _max_entries_limit(dataset_entry.filters)
        accepted_count = 0
        extracted = _extract_dataset_entry(dataset_entry)
        for doc in extracted:
            if _passes_filters(doc, dataset_entry.filters):
                if max_entries is not None and accepted_count >= max_entries:
                    break
                docs.append(doc)
                accepted_count += 1
    return docs


def _extract_dataset_entry(dataset_entry: DatasetEntrySpec) -> list[ExtractedDocument]:
    source = dataset_entry.source
    if source.source_type == "local_text_glob":
        return _extract_local_text_glob(dataset_entry)
    if source.source_type == "huggingface":
        return _extract_huggingface(dataset_entry)
    raise SourceExtractionError(f"Unsupported source type '{source.source_type}' for dataset '{dataset_entry.name}'.")


def _extract_local_text_glob(dataset_entry: DatasetEntrySpec) -> list[ExtractedDocument]:
    mapping = _split_mapping(dataset_entry)
    rows: list[ExtractedDocument] = []
    paths = sorted(glob.glob(dataset_entry.source.uri))
    for index, file_path in enumerate(paths):
        path = Path(file_path)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        source_split = dataset_entry.source.attributes.get("split", "train")
        target_split = mapping.get(source_split, source_split)
        rows.append(
            ExtractedDocument(
                document_id=f"{dataset_entry.name}-{index}",
                dataset_entry=dataset_entry.name,
                split=target_split,
                text=text,
                byte_length=len(text.encode("utf-8")),
                metadata={"path": str(path)},
            )
        )
    return rows


def _extract_huggingface(dataset_entry: DatasetEntrySpec) -> list[ExtractedDocument]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency optional in test env
        raise SourceExtractionError("huggingface source requires 'datasets' package.") from exc

    source = dataset_entry.source
    split = source.attributes.get("split", "train")
    text_column = source.attributes.get("text_column", "text")
    data = load_dataset(source.uri, split=split)

    mapping = _split_mapping(dataset_entry)
    target_split = mapping.get(split, split)
    rows: list[ExtractedDocument] = []
    for index, record in enumerate(data):
        text = str(record.get(text_column, ""))
        rows.append(
            ExtractedDocument(
                document_id=f"{dataset_entry.name}-{index}",
                dataset_entry=dataset_entry.name,
                split=target_split,
                text=text,
                byte_length=len(text.encode("utf-8")),
                metadata={"hf_split": split},
            )
        )
    return rows


def _split_mapping(dataset_entry: DatasetEntrySpec) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if dataset_entry.split_mapping is None:
        return mapping
    for entry in dataset_entry.split_mapping.entries:
        mapping[entry.from_split] = entry.to_split
    return mapping


def _passes_filters(doc: ExtractedDocument, filters: Iterable[FilterSpec]) -> bool:
    for filter_spec in filters:
        filter_type = filter_spec.attributes.get("type")
        value = filter_spec.attributes.get("value")
        if filter_type == "min_bytes" and value is not None and doc.byte_length < int(value):
            return False
        if filter_type == "max_bytes" and value is not None and doc.byte_length > int(value):
            return False
        if filter_type == "contains" and value is not None and value not in doc.text:
            return False
    return True


def _max_entries_limit(filters: Iterable[FilterSpec]) -> int | None:
    for filter_spec in filters:
        if filter_spec.attributes.get("type") == "max_entries":
            value = filter_spec.attributes.get("value")
            if value is not None:
                return int(value)
    return None
