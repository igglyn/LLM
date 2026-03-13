from __future__ import annotations

import random
from collections import defaultdict

from shared.config.specs import ResolvedConfigSpec

from .schemas import MixtureEntry, NormalizedDocument


class MixtureError(ValueError):
    pass


def run_mixture(config: ResolvedConfigSpec, docs: list[NormalizedDocument]) -> list[MixtureEntry]:
    grouped: dict[str, list[NormalizedDocument]] = defaultdict(list)
    for doc in docs:
        grouped[doc.dataset_entry].append(doc)

    mixture_spec = config.dataset.mixture_build
    randomizer = random.Random(int(mixture_spec.attributes.get("random_seed", "0")))
    for per_dataset in grouped.values():
        randomizer.shuffle(per_dataset)

    target = int(mixture_spec.attributes.get("target_documents", str(sum(len(v) for v in grouped.values()))))
    selected: list[MixtureEntry] = []

    for group in mixture_spec.groups:
        pool: list[NormalizedDocument] = []
        for dataset_ref in group.dataset_refs:
            pool.extend(grouped.get(dataset_ref.name, []))

        group_target = int(round((group.percentage / 100.0) * target))
        for doc in pool[:group_target]:
            selected.append(
                MixtureEntry(
                    document_id=doc.document_id,
                    group=group.name,
                    dataset_entry=doc.dataset_entry,
                    split=doc.split,
                    text=doc.text,
                    byte_length=doc.byte_length,
                )
            )

    if len(selected) < target:
        raise MixtureError("Mixture underfilled target_documents; adjust percentages or target.")
    return selected[:target]
