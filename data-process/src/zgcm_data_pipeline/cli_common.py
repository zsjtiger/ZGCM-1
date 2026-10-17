"""Shared argument parsing and summary handling for the command-line entry points.

Every entry point accepts the same input, output, configuration and manifest
options. Only the selection of stage and category, and the pipeline it runs,
differ between them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


def build_base_parser(description: str, *, profiles: Iterable[str] | None = None) -> argparse.ArgumentParser:
    """Parser with the options common to all entry points.

    ``profiles`` adds ``--dataset-profile`` with those names as the only choices.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Summary JSON path; defaults to <output>.summary.json")
    parser.add_argument("--dataset", required=True)
    if profiles is not None:
        parser.add_argument("--dataset-profile", choices=sorted(profiles), help="Optional dataset-specific cleaning profile")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-fields", default="text", help="Comma-separated text fields")
    parser.add_argument("--config", type=Path, help="JSON object with quality, dedup and length policies")
    parser.add_argument("--require-license", action="store_true", help="Mark records without license metadata for review")
    parser.add_argument("--manifest", type=Path, help="Write a release manifest next to the cleaned JSONL")
    parser.add_argument("--keep-dropped", action="store_true")
    return parser


def load_config(path: Path | None, *, require_license: bool) -> dict[str, object]:
    config: dict[str, object] = {}
    if path is not None:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--config must contain one JSON object")
        config.update(loaded)
    if require_license:
        config["require_license"] = True
    return config


def parse_text_fields(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def write_summary(summary: Mapping[str, Any], output: Path, summary_path: Path | None) -> Path:
    path = summary_path or output.with_suffix(output.suffix + ".summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_stage_cli(args: argparse.Namespace, run: Callable[..., dict[str, Any]], **selection: Any) -> int:
    """Run a stage's ``run_jsonl`` from parsed arguments and write its summary."""
    summary = run(
        args.input,
        args.output,
        dataset=args.dataset,
        dataset_profile=getattr(args, "dataset_profile", None),
        id_field=args.id_field,
        text_fields=parse_text_fields(args.text_fields),
        config=load_config(args.config, require_license=args.require_license),
        include_dropped=args.keep_dropped,
        manifest_path=args.manifest,
        **selection,
    )
    write_summary(summary, args.output, args.summary)
    return 0


__all__ = [
    "build_base_parser",
    "load_config",
    "parse_text_fields",
    "run_stage_cli",
    "write_summary",
]
