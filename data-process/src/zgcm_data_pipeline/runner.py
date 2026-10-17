"""File-to-file runner shared by every stage entry point.

A stage supplies a built :class:`~.core.Pipeline` and the labels that identify
it; this module handles adaptation, streaming, writing, the summary and the
manifest, so the stages cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adapters import DatasetAdapter
from .core import Pipeline, PipelineContext
from .models import DataCategory, Stage
from .storage.jsonl import iter_jsonl, write_jsonl
from .storage.manifest import build_manifest, write_manifest
from .storage.release import release_record


def run_pipeline_file(
    pipeline: Pipeline,
    *,
    stage: Stage,
    category: DataCategory,
    dataset: str,
    input_path: Path,
    output_path: Path,
    id_field: str = "id",
    text_fields: tuple[str, ...] = ("text",),
    config: Mapping[str, object] | None = None,
    dataset_profile: str | None = None,
    include_dropped: bool = False,
    manifest_path: Path | None = None,
    summary_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Clean ``input_path`` with ``pipeline`` and write ``output_path``.

    ``summary_fields`` carries stage-specific labels (for example ``source`` for
    Pretrain or ``category`` for the other stages) into the returned summary.
    """
    adapter = DatasetAdapter(
        dataset=dataset,
        stage=stage,
        category=category,
        id_field=id_field,
        text_fields=text_fields,
    )
    context = PipelineContext(config=dict(config or {}))
    cleaned = pipeline.run(adapter.adapt_many(iter_jsonl(input_path)), context, include_dropped=include_dropped)
    written = write_jsonl(output_path, (release_record(sample) for sample in cleaned))
    summary: dict[str, Any] = {
        **dict(summary_fields or {}),
        "stage": stage.value,
        "dataset": dataset,
        "dataset_profile": dataset_profile,
        "written": written,
        "pipeline": pipeline.step_names(),
        "counters": dict(context.counters),
    }
    if manifest_path is not None:
        # Build the manifest from the written file so it matches what consumers read.
        write_manifest(
            manifest_path,
            build_manifest(iter_jsonl(output_path), stage=stage.value, output=output_path.name),
        )
        summary["manifest"] = manifest_path.name
    return summary


__all__ = ["run_pipeline_file"]
