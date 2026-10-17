"""Web/Knowledge classifier-score and token-length bucketing.

Quality bands, length boundaries and the minimum band to keep come from the
``web_knowledge`` profile (``profiles/web_knowledge.json``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....core import PipelineContext, Step
from ....models import Decision, Sample
from ....profiles import resolve_profile


def _params(params: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return params if params is not None else resolve_profile({}, "midtrain", "web_knowledge")


def quality_band(score: float, params: Mapping[str, Any] | None = None) -> str:
    """Label of the highest band whose minimum the score reaches; the default band otherwise."""
    resolved = _params(params)
    for band in sorted(resolved["quality_bands"], key=lambda item: float(item["min"]), reverse=True):
        if score >= float(band["min"]):
            return str(band["label"])
    return str(resolved["default_band"])


def length_bucket(token_count: int, params: Mapping[str, Any] | None = None) -> str | None:
    """Length label for a positive token count; ``None`` when the count is not positive."""
    if token_count <= 0:
        return None
    resolved = _params(params)
    for upper, label in resolved["length_boundaries"]:
        if token_count <= int(upper):
            return str(label)
    return str(resolved["overflow_length_label"])


def _value(data: dict[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    metadata = data.get("meta") if isinstance(data.get("meta"), dict) else data.get("metadata")
    return metadata.get(key) if isinstance(metadata, dict) else None


def clean_web_knowledge_record(sample: Sample, context: PipelineContext) -> Sample:
    """Attach the quality band and length bucket, then drop records below the minimum band."""
    params = resolve_profile(context.config, "midtrain", "web_knowledge")
    score = _value(sample.data, "quality_mean")
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = -1.0
        sample.data["quality_missing"] = True
        sample.flags.add("web_knowledge_missing_quality_score")
        sample.mark(Decision.REVIEW, "web_knowledge_missing_quality_score")
    else:
        sample.data["quality_missing"] = False
    band = quality_band(score_value, params)
    sample.data["quality_mean"] = score_value
    sample.data["quality_band"] = band
    sample.metrics["web_knowledge_quality_mean"] = score_value
    sample.buckets["classifier_quality"] = band

    token_count = _value(sample.data, "token_count")
    try:
        token_value = int(token_count)
    except (TypeError, ValueError):
        token_value = 0
    bucket = length_bucket(token_value, params)
    if bucket is None:
        sample.flags.add("web_knowledge_missing_token_count")
        sample.mark(Decision.REVIEW, "web_knowledge_missing_token_count")
    else:
        sample.metrics["training_token_count"] = float(token_value)
        sample.buckets["source_length"] = bucket

    # Bands are ordered from the default (lowest) upward.
    labels = [str(params["default_band"]), *(str(band["label"]) for band in sorted(params["quality_bands"], key=lambda item: float(item["min"])))]
    order = {label: index for index, label in enumerate(labels)}
    minimum_band = str(params["min_quality_band"])
    if minimum_band not in order:
        raise ValueError(f"web_knowledge_min_quality_band must be one of {', '.join(labels)}")
    if order[band] < order[minimum_band]:
        sample.mark(Decision.DROP, "web_knowledge_below_quality_band")
    return sample


WEB_KNOWLEDGE_STEPS = (
    Step(
        "clean_web_knowledge_record",
        clean_web_knowledge_record,
        "Apply Web/Knowledge classifier quality and length buckets",
    ),
)


__all__ = ["WEB_KNOWLEDGE_STEPS", "clean_web_knowledge_record", "length_bucket", "quality_band"]
