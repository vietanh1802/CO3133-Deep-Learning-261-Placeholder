"""Keep foundational packages independent of their higher-level consumers."""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src"


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("data", {"eda", "plots", "training", "evaluation", "error"}),
        ("models", {"eda", "plots", "training", "evaluation", "error", "data"}),
        ("config", {"eda", "plots", "training", "evaluation", "error", "data", "models"}),
        (
            "utils",
            {"eda", "plots", "training", "evaluation", "error", "data", "models", "config"},
        ),
    ],
)
def test_foundational_packages_do_not_import_their_consumers(package, forbidden):
    for path in (SOURCE / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parent = ".".join(path.parent.relative_to(SOURCE.parent).parts)
                    module = resolve_name("." * node.level + module, parent)
                modules.append(module)
                modules.extend(f"{module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                parts = module.split(".")
                if len(parts) > 1 and parts[0] == "src":
                    assert parts[1] not in forbidden, f"{path.relative_to(SOURCE)} imports {module}"
