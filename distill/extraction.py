from __future__ import annotations

import glob
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
    if dataset_entry.source.source_type != "local_text_glob":
        raise ExtractionError(f"Unsupported source type '{dataset_entry.source.source_type}'.")

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


def _passes_filters(doc: NormalizedDocument, filters: list[FilterSpec]) -> bool:
    for filter_spec in filters:
        filter_type = filter_spec.attributes.get("type")
        value = filter_spec.attributes.get("value")
        if filter_type == "min_bytes" and value is not None and doc.byte_length < int(value):
            return False
        if filter_type == "max_bytes" and value is not None and doc.byte_length > int(value):
            return False
    return True
