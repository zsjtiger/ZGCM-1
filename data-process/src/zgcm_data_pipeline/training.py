"""Compatibility alias for the split modules.

Sampling lives in :mod:`.sampling`, packing in :mod:`.packing` and work
scheduling in :mod:`.scheduling`. New code should import from those modules.
"""

from .packing import IGNORE_INDEX, PackedSequence, TokenizedSample, build_loss_mask, pack_tokenized_samples, validate_tokenized_sample
from .sampling import allocate_mix_quotas, deterministic_shuffle_key, select_mixed_sample_ids, stable_partition
from .scheduling import RowMicroshard, WorkUnit, balance_work_units, plan_row_microshards

__all__ = [
    "IGNORE_INDEX",
    "PackedSequence",
    "RowMicroshard",
    "TokenizedSample",
    "WorkUnit",
    "allocate_mix_quotas",
    "balance_work_units",
    "build_loss_mask",
    "deterministic_shuffle_key",
    "pack_tokenized_samples",
    "plan_row_microshards",
    "select_mixed_sample_ids",
    "stable_partition",
    "validate_tokenized_sample",
]
