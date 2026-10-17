"""Tokenized-sample packing with SFT loss-mask boundaries preserved.

The tokenizer itself is not part of this package; callers supply token ids
(and, for SFT, per-token targets where ``IGNORE_INDEX`` marks untrained tokens).
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class TokenizedSample:
    tokens: tuple[int, ...]
    source_id: str
    targets: tuple[int, ...] | None = None

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class PackedSequence:
    tokens: list[int] = field(default_factory=list)
    targets: list[int] | None = None
    cu_seqlens: list[int] = field(default_factory=lambda: [0])
    source_ids: list[str] = field(default_factory=list)


def validate_tokenized_sample(sample: TokenizedSample, *, ignore_index: int = IGNORE_INDEX) -> None:
    if not sample.tokens:
        raise ValueError("tokenized sample is empty")
    if sample.targets is None:
        return
    if len(sample.tokens) != len(sample.targets):
        raise ValueError("tokens and targets must have the same length")
    if not any(target != ignore_index for target in sample.targets):
        raise ValueError("SFT sample has no trainable target token")


def build_loss_mask(targets: Sequence[int], *, ignore_index: int = IGNORE_INDEX) -> tuple[int, ...]:
    return tuple(0 if target == ignore_index else 1 for target in targets)


def pack_tokenized_samples(
    samples: Iterable[TokenizedSample],
    *,
    sequence_length: int,
    allow_overlength: bool = False,
) -> tuple[list[PackedSequence], list[TokenizedSample]]:
    """First-fit-decreasing packing while preserving SFT target boundaries."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    accepted: list[TokenizedSample] = []
    overlength: list[TokenizedSample] = []
    for sample in samples:
        validate_tokenized_sample(sample)
        if sample.length > sequence_length:
            overlength.append(sample)
            if not allow_overlength:
                continue
        accepted.append(sample)

    accepted.sort(key=lambda sample: (-sample.length, sample.source_id))
    bins: list[PackedSequence] = []
    free: list[tuple[int, int]] = []
    for sample in accepted:
        position = bisect.bisect_left(free, (sample.length, -1))
        if position == len(free):
            bin_id = len(bins)
            bins.append(PackedSequence(targets=[] if sample.targets is not None else None))
        else:
            _, bin_id = free.pop(position)
        packed = bins[bin_id]
        if (packed.targets is None) != (sample.targets is None):
            raise ValueError("cannot mix pretraining and SFT samples in one packing call")
        packed.tokens.extend(sample.tokens)
        if packed.targets is not None and sample.targets is not None:
            packed.targets.extend(sample.targets)
        packed.source_ids.append(sample.source_id)
        packed.cu_seqlens.append(len(packed.tokens))
        remaining = sequence_length - len(packed.tokens)
        if remaining > 0:
            bisect.insort(free, (remaining, bin_id))
    return bins, overlength


__all__ = [
    "IGNORE_INDEX",
    "PackedSequence",
    "TokenizedSample",
    "build_loss_mask",
    "pack_tokenized_samples",
    "validate_tokenized_sample",
]
