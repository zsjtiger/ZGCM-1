"""Compatibility alias for :mod:`zgcm_data_pipeline.sampling` (prefix-sampled mixes)."""

from ...sampling import DEFAULT_MIX_ELIGIBLE as DEFAULT_ELIGIBLE
from ...sampling import prefix_sample_mix

__all__ = ["DEFAULT_ELIGIBLE", "prefix_sample_mix"]
