"""OLMOCR science-PDF compression-ratio filtering."""

from __future__ import annotations

import zlib

from ....core import PipelineContext, Step
from ....profiles import resolve_profile
from ....models import Decision, Sample


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="ignore")
    return len(zlib.compress(raw, 6)) / len(raw) if raw else 0.0


def clean_olmocr_record(sample: Sample, context: PipelineContext) -> Sample:
    """Remove highly compressible OCR artifacts using the archived 0.08 rule."""
    data = sample.data
    text = data.get("text") if isinstance(data.get("text"), str) else data.get("content")
    if not isinstance(text, str) or not text.strip():
        sample.mark(Decision.DROP, "olmocr_missing_text")
        return sample
    data["text"] = text
    sample.text_fields = ("text",)
    ratio = compression_ratio(text)
    sample.metrics["olmocr_zlib_ratio"] = ratio
    params = resolve_profile(context.config, "pretrain", "olmocr")
    threshold = float(params["min_compression_ratio"])
    if ratio < threshold:
        sample.mark(Decision.DROP, "olmocr_low_compression_ratio")
    score = data.get("edu_score")
    if score is not None:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            sample.mark(Decision.REVIEW, "olmocr_invalid_edu_score")
        else:
            sample.metrics["olmocr_edu_score"] = score_value
            if score_value < float(params["min_edu_score"]):
                sample.mark(Decision.DROP, "olmocr_low_edu_score")
    sample.buckets["olmocr_policy"] = "compression_ratio_ge_0p08"
    return sample


OLMOCR_STEPS = (
    Step("clean_olmocr_record", clean_olmocr_record, "Filter OLMOCR compression artifacts"),
)


__all__ = ["OLMOCR_STEPS", "clean_olmocr_record", "compression_ratio"]
