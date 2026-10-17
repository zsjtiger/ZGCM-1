"""Compatibility alias for :mod:`zgcm_data_pipeline.algorithms.simhash`."""

from .algorithms.simhash import NearDeduper, hamming_distance, simhash64

__all__ = ["NearDeduper", "hamming_distance", "simhash64"]
