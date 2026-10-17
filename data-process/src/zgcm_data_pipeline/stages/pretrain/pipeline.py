"""Environment-neutral Pretrain pipeline entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...core import Pipeline
from ...models import DataCategory, Stage
from ...runner import run_pipeline_file
from ...steps.common import COMMON_PREFIX, COMMON_SUFFIX
from .datasets import dataset_steps
from .source_cleaning import PRETRAIN_CODE_STEPS, PRETRAIN_PDF_STEPS, PRETRAIN_TEXT_STEPS


SOURCE_CATEGORIES = {
    "code": DataCategory.CODE,
    "pdf_ocr": DataCategory.PDF_OCR,
    "general_text": DataCategory.GENERAL_TEXT,
}


def build_pipeline(source: str, dataset_profile: str | None = None) -> Pipeline:
    """Build a Pretrain pipeline for one of the three supported source families."""
    key = str(source).strip().lower()
    if key not in SOURCE_CATEGORIES:
        raise ValueError(f"unsupported pretrain source: {source!r}; choose from {sorted(SOURCE_CATEGORIES)}")
    if key == "code":
        source_steps = PRETRAIN_CODE_STEPS
    elif key == "pdf_ocr":
        source_steps = PRETRAIN_PDF_STEPS
    else:
        source_steps = PRETRAIN_TEXT_STEPS
    return Pipeline(
        name=f"pretrain_{key}_{dataset_profile or 'generic'}_pipeline",
        steps=dataset_steps(dataset_profile, key) + source_steps + COMMON_PREFIX + COMMON_SUFFIX,
    )


def run_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    dataset: str,
    source: str,
    id_field: str = "id",
    text_fields: tuple[str, ...] = ("text",),
    config: Mapping[str, object] | None = None,
    dataset_profile: str | None = None,
    include_dropped: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run a Pretrain source pipeline without assuming any local environment."""
    key = str(source).strip().lower()
    category = SOURCE_CATEGORIES.get(key)
    if category is None:
        raise ValueError(f"unsupported pretrain source: {source!r}")
    return run_pipeline_file(
        build_pipeline(key, dataset_profile),
        stage=Stage.PRETRAIN,
        category=category,
        dataset=dataset,
        input_path=input_path,
        output_path=output_path,
        id_field=id_field,
        text_fields=text_fields,
        config=config,
        dataset_profile=dataset_profile,
        include_dropped=include_dropped,
        manifest_path=manifest_path,
        summary_fields={"source": key},
    )


__all__ = ["SOURCE_CATEGORIES", "build_pipeline", "run_jsonl"]
