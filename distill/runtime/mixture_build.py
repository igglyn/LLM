from __future__ import annotations

import random
from collections import defaultdict

from shared.config.specs import ConfigSpec

from .types import ExtractedDocument, MixtureDocument


class MixtureBuildError(ValueError):
    pass


def run_mixture_build(config: ConfigSpec, extracted_docs: list[ExtractedDocument]) -> list[MixtureDocument]:
    mixture_spec = config.dataset.mixture_build
    grouped_by_dataset: dict[str, list[ExtractedDocument]] = defaultdict(list)
    for doc in extracted_docs:
        grouped_by_dataset[doc.dataset_entry].append(doc)

    min_bytes = _optional_int(mixture_spec.attributes.get("min_bytes"))
    max_bytes = _optional_int(mixture_spec.attributes.get("max_bytes"))
    filtered_docs = {
        name: [doc for doc in docs if _byte_range_ok(doc, min_bytes=min_bytes, max_bytes=max_bytes)]
        for name, docs in grouped_by_dataset.items()
    }

    random_seed = int(mixture_spec.attributes.get("random_seed", "0"))
    randomizer = random.Random(random_seed)
    for docs in filtered_docs.values():
        randomizer.shuffle(docs)

    target_documents = int(mixture_spec.attributes.get("target_documents", str(sum(len(d) for d in filtered_docs.values()))))

    group_pools: dict[str, list[ExtractedDocument]] = {}
    group_targets: dict[str, int] = {}
    for group in mixture_spec.groups:
        pool: list[ExtractedDocument] = []
        for dataset_ref in group.dataset_refs:
            pool.extend(filtered_docs.get(dataset_ref.name, []))
        group_pools[group.name] = pool
        group_targets[group.name] = int(round((group.percentage / 100.0) * target_documents))

    selected = _select_documents(group_pools, group_targets, target_documents, depletion_policy=mixture_spec.attributes.get("depletion_policy", "rebalance"))

    output: list[MixtureDocument] = []
    for group_name, doc in selected:
        output.append(
            MixtureDocument(
                document_id=doc.document_id,
                group=group_name,
                dataset_entry=doc.dataset_entry,
                split=doc.split,
                text=doc.text,
                byte_length=doc.byte_length,
            )
        )
    return output


def _select_documents(
    group_pools: dict[str, list[ExtractedDocument]],
    group_targets: dict[str, int],
    target_documents: int,
    depletion_policy: str,
) -> list[tuple[str, ExtractedDocument]]:
    selected: list[tuple[str, ExtractedDocument]] = []
    pool_index = {group: 0 for group in group_pools}

    for group_name, target in group_targets.items():
        pool = group_pools[group_name]
        take = min(target, len(pool))
        for _ in range(take):
            selected.append((group_name, pool[pool_index[group_name]]))
            pool_index[group_name] += 1

    if len(selected) >= target_documents:
        return selected[:target_documents]

    if depletion_policy != "rebalance":
        raise MixtureBuildError(f"Unsupported depletion_policy '{depletion_policy}'.")

    # rebalance: round-robin remaining non-depleted pools until target reached
    active_groups = [group for group, pool in group_pools.items() if pool_index[group] < len(pool)]
    while len(selected) < target_documents and active_groups:
        next_active: list[str] = []
        for group_name in active_groups:
            if len(selected) >= target_documents:
                break
            pool = group_pools[group_name]
            index = pool_index[group_name]
            if index < len(pool):
                selected.append((group_name, pool[index]))
                pool_index[group_name] += 1
            if pool_index[group_name] < len(pool):
                next_active.append(group_name)
        active_groups = next_active

    return selected


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _byte_range_ok(doc: ExtractedDocument, min_bytes: int | None, max_bytes: int | None) -> bool:
    if min_bytes is not None and doc.byte_length < min_bytes:
        return False
    if max_bytes is not None and doc.byte_length > max_bytes:
        return False
    return True
