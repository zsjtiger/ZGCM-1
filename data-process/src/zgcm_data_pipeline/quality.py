"""Compatibility alias for :mod:`zgcm_data_pipeline.steps.quality`."""

from .steps.quality import QualityPolicy, apply_quality_policy, quality_policy_key, resolve_quality_policy

__all__ = ["QualityPolicy", "apply_quality_policy", "quality_policy_key", "resolve_quality_policy"]
