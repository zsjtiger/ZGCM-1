"""Enforce the package layering: lower layers never import higher ones.

    models, core                 (kernel)
    storage, algorithms,         (leaf utilities; depend on kernel at most)
    sampling, packing, scheduling
    steps                        (category and shared steps)
    pipelines, runner, adapters  (composition)
    stages, cli                  (entry points)
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "zgcm_data_pipeline"
PACKAGE_NAME = "zgcm_data_pipeline"

FORBIDDEN_IMPORTS = {
    "models": {"core", "storage", "algorithms", "sampling", "packing", "scheduling", "steps", "pipelines", "runner", "adapters", "stages", "cli"},
    "core": {"storage", "algorithms", "sampling", "packing", "scheduling", "steps", "pipelines", "runner", "adapters", "stages", "cli"},
    "storage": {"algorithms", "sampling", "packing", "scheduling", "steps", "pipelines", "runner", "adapters", "stages", "cli"},
    "algorithms": {"core", "storage", "sampling", "packing", "scheduling", "steps", "pipelines", "runner", "adapters", "stages", "cli"},
    "sampling": {"core", "storage", "algorithms", "packing", "scheduling", "steps", "pipelines", "runner", "adapters", "stages", "cli"},
    "packing": {"core", "storage", "algorithms", "sampling", "scheduling", "steps", "pipelines", "runner", "adapters", "stages", "cli"},
    "scheduling": {"core", "storage", "algorithms", "sampling", "packing", "steps", "pipelines", "runner", "adapters", "stages", "cli"},
    "steps": {"pipelines", "runner", "adapters", "stages", "cli"},
}

# Top-level compatibility aliases may import from the new layout; they are the
# only modules allowed to sit outside the layer they forward to.
COMPATIBILITY_ALIASES = {"io", "dedup", "quality", "training"}


def layer_of(path: Path) -> str | None:
    """Return the layer name for a module path inside the package, if it has one."""
    relative = path.relative_to(PACKAGE)
    if len(relative.parts) == 1:
        name = relative.stem
        return None if name in COMPATIBILITY_ALIASES | {"__init__", "__main__"} else name
    if relative.parts[0] == "stages":
        return "stages"
    return relative.parts[0]


def imported_layers(path: Path) -> set[str]:
    """Layers reached by relative imports in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(PACKAGE).with_suffix("")
    package_parts = list(relative.parts[:-1]) if path.name != "__init__.py" else list(relative.parts[:-1])
    layers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        base = package_parts[: len(package_parts) - (node.level - 1)] if node.level > 1 else package_parts
        target = base + (node.module.split(".") if node.module else [])
        if not target:
            # ``from . import x``: every name is a sibling module of this package.
            layers.update(alias.name for alias in node.names)
            continue
        layers.add(target[0] if target[0] != "stages" else "stages")
    return layers


class LayeringTests(unittest.TestCase):
    def test_lower_layers_do_not_import_higher_layers(self) -> None:
        violations: list[str] = []
        for path in sorted(PACKAGE.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            layer = layer_of(path)
            if layer not in FORBIDDEN_IMPORTS:
                continue
            forbidden = FORBIDDEN_IMPORTS[layer] & imported_layers(path)
            if forbidden:
                violations.append(f"{path.relative_to(PACKAGE)} ({layer}) imports {sorted(forbidden)}")
        self.assertEqual(violations, [], "layering violations:\n" + "\n".join(violations))

    def test_no_stage_package_is_imported_by_a_kernel_module(self) -> None:
        # Kept explicit because it is the regression that motivated the layout:
        # shared algorithms used to live under stages/midtrain.
        for path in PACKAGE.rglob("*.py"):
            if "__pycache__" in path.parts or layer_of(path) in {"stages", "cli"}:
                continue
            self.assertNotIn("stages", imported_layers(path), f"{path.relative_to(PACKAGE)} imports stages")


if __name__ == "__main__":
    unittest.main()
