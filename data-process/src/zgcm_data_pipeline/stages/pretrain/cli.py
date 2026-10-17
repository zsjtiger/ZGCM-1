"""Command-line entry point for Pretrain source cleaning."""

from __future__ import annotations

import argparse

from ...cli_common import build_base_parser, run_stage_cli
from .datasets import DATASET_PROFILES
from .pipeline import SOURCE_CATEGORIES, run_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser(
        "Run an environment-neutral ZGCM-1 Pretrain source pipeline",
        profiles=DATASET_PROFILES,
    )
    parser.add_argument("--source", choices=sorted(SOURCE_CATEGORIES), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_stage_cli(args, run_jsonl, source=args.source)


if __name__ == "__main__":
    raise SystemExit(main())
