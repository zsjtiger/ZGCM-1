"""Environment-neutral Midtrain category pipeline entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...core import Pipeline
from ...models import DataCategory, Stage
from ...pipelines import build_pipeline as build_category_pipeline
from ...runner import run_pipeline_file
from .datasets import dataset_steps


MIDTRAIN_CATEGORIES = tuple(category.value for category in DataCategory if category.value in {"code", "web", "instruction", "agentic", "math", "reasoning"})


def build_pipeline(category: str, dataset_profile: str | None = None) -> Pipeline:
    """Build the normalized Midtrain pipeline for one supported category."""
    key = str(category).strip().lower()
    if key not in MIDTRAIN_CATEGORIES:
        raise ValueError(f"unsupported midtrain category: {category!r}; choose from {MIDTRAIN_CATEGORIES}")
    category_pipeline = build_category_pipeline(DataCategory(key))
    profile_steps = dataset_steps(dataset_profile, key)
    if not profile_steps:
        return category_pipeline
    return Pipeline(name=f"midtrain_{key}_{dataset_profile}_pipeline", steps=profile_steps + category_pipeline.steps)


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
    """Run a Midtrain category pipeline using only explicit file/config arguments."""
    key = str(category).strip().lower()
    return run_pipeline_file(
        build_pipeline(key, dataset_profile),
        stage=Stage.MIDTRAIN,
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


__all__ = ["MIDTRAIN_CATEGORIES", "build_pipeline", "run_jsonl"]
