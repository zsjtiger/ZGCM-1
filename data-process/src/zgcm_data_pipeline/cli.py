"""Generic category CLI: ``zgcm-clean``.

Runs one category pipeline with an explicit ``--stage`` label. Stage-specific
entry points (``zgcm-pretrain-clean`` and friends) also accept dataset profiles.
"""

from __future__ import annotations

import argparse

from .cli_common import build_base_parser, load_config, parse_text_fields, write_summary
from .models import DataCategory, Stage
from .pipelines import build_pipeline
from .runner import run_pipeline_file


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser("Run a ZGCM-1 category cleaning pipeline")
    parser.add_argument("--stage", choices=[item.value for item in Stage], required=True)
    parser.add_argument("--category", choices=[item.value for item in DataCategory], required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    category = DataCategory(args.category)
    summary = run_pipeline_file(
        build_pipeline(category),
        stage=Stage(args.stage),
        category=category,
        dataset=args.dataset,
        input_path=args.input,
        output_path=args.output,
        id_field=args.id_field,
        text_fields=parse_text_fields(args.text_fields),
        config=load_config(args.config, require_license=args.require_license),
        include_dropped=args.keep_dropped,
        manifest_path=args.manifest,
        summary_fields={"category": args.category},
    )
    write_summary(summary, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
