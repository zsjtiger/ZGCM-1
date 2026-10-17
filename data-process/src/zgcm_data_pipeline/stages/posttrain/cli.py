"""Command-line entry point for Posttrain (SFT) cleaning."""

from __future__ import annotations

import argparse

from ...cli_common import build_base_parser, run_stage_cli
from .datasets import DATASET_PROFILES
from .pipeline import POSTTRAIN_CATEGORIES, run_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser(
        "Run an environment-neutral ZGCM-1 Posttrain pipeline",
        profiles=DATASET_PROFILES,
    )
    parser.add_argument("--category", choices=sorted(POSTTRAIN_CATEGORIES), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_stage_cli(args, run_jsonl, category=args.category)


if __name__ == "__main__":
    raise SystemExit(main())
