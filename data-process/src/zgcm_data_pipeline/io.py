"""Compatibility alias for :mod:`zgcm_data_pipeline.storage.jsonl`."""

from .storage.jsonl import iter_jsonl, open_text, write_jsonl

__all__ = ["iter_jsonl", "open_text", "write_jsonl"]
