"""Deterministic partitioning, quota allocation and sample selection.

Everything here is pure: the same inputs and seed always give the same output,
and nothing depends on stage, dataset or filesystem layout.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence


DEFAULT_MIX_ELIGIBLE = {
    "mix16": ("B16",),
    "mix64": ("B16", "B64"),
    "mix256": ("B16", "B64", "B256"),
}


def deterministic_shuffle_key(record_key: str, *, seed: str = "zgcm") -> bytes:
    """Return a stable sort key for reproducible sample-level shuffling."""
    return hashlib.blake2b(f"{seed}\0{record_key}".encode("utf-8"), digest_size=16).digest()


def stable_partition(record_key: str, partitions: int, *, seed: str = "zgcm") -> int:
    """Place one sample in a deterministic shuffle partition."""
    if partitions <= 0:
        raise ValueError("partitions must be positive")
    digest = hashlib.blake2b(f"{seed}\0{record_key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % partitions


def allocate_mix_quotas(weights: Mapping[str, float], total_samples: int) -> dict[str, int]:
    """Allocate an exact sample budget with normalized weights and largest remainders."""
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    clean = {str(name): float(weight) for name, weight in weights.items() if float(weight) > 0}
    if not clean:
        raise ValueError("at least one positive mix weight is required")
    total_weight = sum(clean.values())
    exact = {name: total_samples * weight / total_weight for name, weight in clean.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total_samples - sum(quotas.values())
    order = sorted(clean, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def select_mixed_sample_ids(
    source_to_ids: Mapping[str, Iterable[str]],
    quotas: Mapping[str, int],
    *,
    seed: str = "zgcm",
) -> list[str]:
    """Select each source quota and deterministically interleave the resulting mix."""
    selected: list[tuple[bytes, str]] = []
    for source, quota in sorted(quotas.items()):
        if quota < 0:
            raise ValueError("mix quotas must be non-negative")
        ranked = sorted(
            (str(record_id) for record_id in source_to_ids.get(source, ())),
            key=lambda record_id: deterministic_shuffle_key(record_id, seed=f"{seed}:{source}"),
        )
        if len(ranked) < quota:
            raise ValueError(f"source {source!r} has {len(ranked)} samples but quota is {quota}")
        for record_id in ranked[:quota]:
            selected.append((deterministic_shuffle_key(record_id, seed=f"{seed}:mix"), record_id))
    selected.sort(key=lambda item: (item[0], item[1]))
    return [record_id for _, record_id in selected]


def prefix_sample_mix(
    base_buckets: Mapping[str, Sequence[str]],
    final_targets: Mapping[str, int],
    *,
    seed: str = "zgcm-midtrain",
    eligible_by_final: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Select disjoint final mixes from nested prefix-eligible base buckets.

    ``final_targets`` is processed in the canonical order ``mix16``,
    ``mix64``, ``mix256``. A custom ``eligible_by_final`` can be supplied for
    another stage configuration, but each final mix still excludes all IDs
    selected by earlier mixes.
    """
    eligibility = eligible_by_final or DEFAULT_MIX_ELIGIBLE
    selected: set[str] = set()
    result: dict[str, list[str]] = {}
    order = [name for name in ("mix16", "mix64", "mix256") if name in final_targets]
    order.extend(name for name in final_targets if name not in order)
    for final_name in order:
        target = int(final_targets[final_name])
        if target < 0:
            raise ValueError("final mix targets must be non-negative")
        candidates = {
            str(record_id)
            for bucket in eligibility.get(final_name, ())
            for record_id in base_buckets.get(bucket, ())
            if str(record_id) not in selected
        }
        if len(candidates) < target:
            raise ValueError(f"{final_name} needs {target} samples but only {len(candidates)} are available")
        chosen = sorted(
            candidates,
            key=lambda record_id: (hashlib.blake2b(f"{seed}:{final_name}\0{record_id}".encode("utf-8"), digest_size=16).digest(), record_id),
        )[:target]
        result[final_name] = chosen
        selected.update(chosen)
    return result


__all__ = [
    "DEFAULT_MIX_ELIGIBLE",
    "allocate_mix_quotas",
    "deterministic_shuffle_key",
    "prefix_sample_mix",
    "select_mixed_sample_ids",
    "stable_partition",
]
