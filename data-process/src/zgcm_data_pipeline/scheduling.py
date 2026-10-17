"""Work-unit balancing and contiguous row-span planning for parallel jobs."""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkUnit:
    key: str
    weight: int


@dataclass(frozen=True, slots=True)
class RowMicroshard:
    source: str
    row_start: int
    row_end: int
    tokens: int


def balance_work_units(units: Sequence[WorkUnit], workers: int) -> list[list[WorkUnit]]:
    """Assign heavy units first to the currently lightest worker (LPT)."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    groups: list[list[WorkUnit]] = [[] for _ in range(workers)]
    loads = [(0, worker) for worker in range(workers)]
    for unit in sorted(units, key=lambda item: (-item.weight, item.key)):
        load, worker = loads.pop(0)
        groups[worker].append(unit)
        bisect.insort(loads, (load + max(0, int(unit.weight)), worker))
    return groups


def plan_row_microshards(
    source: str,
    row_token_counts: Iterable[int],
    *,
    target_tokens: int,
) -> list[RowMicroshard]:
    """Build contiguous row spans close to a target token budget."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    plans: list[RowMicroshard] = []
    start = 0
    total = 0
    end = 0
    for end, raw_tokens in enumerate(row_token_counts, 1):
        tokens = max(0, int(raw_tokens))
        if total and total + tokens > target_tokens:
            plans.append(RowMicroshard(source, start, end - 1, total))
            start = end - 1
            total = 0
        total += tokens
    if end > start:
        plans.append(RowMicroshard(source, start, end, total))
    return plans


__all__ = ["RowMicroshard", "WorkUnit", "balance_work_units", "plan_row_microshards"]
