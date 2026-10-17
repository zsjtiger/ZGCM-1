"""Environment-neutral Posttrain (SFT) pipeline entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...core import Pipeline
from ...models import DataCategory, Stage
from ...pipelines import build_pipeline as build_category_pipeline
from ...runner import run_pipeline_file
from .datasets import dataset_steps
from .format import POSTTRAIN_FORMAT_STEPS


POSTTRAIN_CATEGORIES = ("instruction", "agentic", "math", "reasoning", "code", "web")


def build_pipeline(category: str, dataset_profile: str | None = None) -> Pipeline:
    """Build a Posttrain pipeline with format normalization before shared checks."""
    key = str(category).strip().lower()
    if key not in POSTTRAIN_CATEGORIES:
        raise ValueError(f"unsupported posttrain category: {category!r}; choose from {POSTTRAIN_CATEGORIES}")
    category_pipeline = build_category_pipeline(DataCategory(key))
    profile_steps = dataset_steps(dataset_profile, key)
    return Pipeline(
        name=f"posttrain_{key}_{dataset_profile or 'generic'}_pipeline",
        steps=profile_steps + POSTTRAIN_FORMAT_STEPS + category_pipeline.steps,
    )


def run_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    dataset: str,
    category: str,
    id_field: str = "id",
    text_fields: tuple[str, ...] = ("text",),
    config: Mapping[str, object] | None = None,
    dataset_profile: str | None = None,
    include_dropped: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run Posttrain format and quality processing with explicit inputs only."""
    key = str(category).strip().lower()
    return run_pipeline_file(
        build_pipeline(key, dataset_profile),
        stage=Stage.POSTTRAIN,
        category=DataCategory(key),
        dataset=dataset,
        input_path=input_path,
        output_path=output_path,
        id_field=id_field,
        text_fields=text_fields,
        config=config,
        dataset_profile=dataset_profile,
        include_dropped=include_dropped,
        manifest_path=manifest_path,
        summary_fields={"category": key},
    )


__all__ = ["POSTTRAIN_CATEGORIES", "build_pipeline", "run_jsonl"]
