"""Serialized release records and their content hashes."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..models import Sample


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def release_record(sample: Sample) -> dict[str, Any]:
    """Stable serialized form used by materializers and manifests."""
    result = sample.as_dict()
    result["record_sha256"] = stable_hash(
        json.dumps(result["data"], ensure_ascii=False, sort_keys=True, default=str)
    )
    return result


__all__ = ["release_record", "stable_hash"]
