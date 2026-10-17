"""Dataset-profile parameters kept as data.

Each profile's defaults live in ``stages/<stage>/profiles/<profile>.json``.
A run can override any parameter with a configuration key named
``<profile>_<key>``, for example ``finepdfs_lid_threshold``; that is the same
naming the profiles already used before the defaults moved into JSON.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


PROFILE_ROOT = Path(__file__).resolve().parent / "stages"


@lru_cache(maxsize=None)
def _load_defaults(stage: str, profile: str) -> Mapping[str, Any]:
    path = PROFILE_ROOT / stage / "profiles" / f"{profile}.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"profile file must contain one JSON object: {path}")
    return loaded


def resolve_profile(config: Mapping[str, object], stage: str, profile: str) -> dict[str, Any]:
    """Return every parameter of ``profile``, with run-config overrides applied.

    The result is a fresh copy, so callers may keep or modify it.
    """
    defaults = _load_defaults(stage, profile)
    resolved: dict[str, Any] = {}
    for key, default in defaults.items():
        name = f"{profile}_{key}"
        resolved[key] = copy.deepcopy(config[name] if name in config else default)
    return resolved


__all__ = ["resolve_profile"]
