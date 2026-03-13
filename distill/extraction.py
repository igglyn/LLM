from __future__ import annotations

import glob
import importlib
from pathlib import Path

from shared.config.specs import DatasetEntrySpec, FilterSpec, ResolvedConfigSpec

from .schemas import NormalizedDocument


class ExtractionError(ValueError):
    pass


def run_extraction(config: ResolvedConfigSpec) -> list[NormalizedDocument]:
    output: list[NormalizedDocument] = []
    for entry in config.dataset.source_extraction.dataset_entries:
        for doc in _extract_entry(entry):
            if _passes_filters(doc, entry.filters):
                output.append(doc)
    return output


def _extract_entry(dataset_entry: DatasetEntrySpec) -> list[NormalizedDocument]:
    source_type = dataset_entry.source.source_type
    if source_type == "local_text_glob":
        return _extract_local_text_glob(dataset_entry)
    if source_type == "huggingface":
        return _extract_huggingface(dataset_entry)
    raise ExtractionError(f"Unsupported source type '{source_type}'.")


def _extract_local_text_glob(dataset_entry: DatasetEntrySpec) -> list[NormalizedDocument]:

    split_map = {
        m.from_split: m.to_split
        for m in (dataset_entry.split_mapping.entries if dataset_entry.split_mapping is not None else [])
    }
    source_split = dataset_entry.source.attributes.get("split", "train")
    target_split = split_map.get(source_split, source_split)

    rows: list[NormalizedDocument] = []
    for index, raw_path in enumerate(sorted(glob.glob(dataset_entry.source.uri))):
        path = Path(raw_path)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            rows.append(
                NormalizedDocument(
                    document_id=f"{dataset_entry.name}-{index}",
                    dataset_entry=dataset_entry.name,
                    split=target_split,
                    text=text,
                    byte_length=len(text.encode("utf-8")),
                    metadata={"path": str(path)},
                )
            )
    return rows


def _extract_huggingface(dataset_entry: DatasetEntrySpec) -> list[NormalizedDocument]:
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ExtractionError("huggingface source requires the optional 'datasets' package.") from exc

    split_map = {
        m.from_split: m.to_split
        for m in (dataset_entry.split_mapping.entries if dataset_entry.split_mapping is not None else [])
    }

    source_split = dataset_entry.source.attributes.get("split", "train")
    text_column = dataset_entry.source.attributes.get("text_column", "text")
    target_split = split_map.get(source_split, source_split)

    dataset_rows = datasets_module.load_dataset(dataset_entry.source.uri, split=source_split)
    rows: list[NormalizedDocument] = []
    for index, raw in enumerate(dataset_rows):
        text = str(raw.get(text_column, ""))
        rows.append(
            NormalizedDocument(
                document_id=f"{dataset_entry.name}-{index}",
                dataset_entry=dataset_entry.name,
                split=target_split,
                text=text,
                byte_length=len(text.encode("utf-8")),
                metadata={"hf_split": source_split, "text_column": text_column},
            )
        )
    return rows


def _passes_filters(doc: NormalizedDocument, filters: list[FilterSpec]) -> bool:
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
