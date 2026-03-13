"""Tests for pages/ore.py module."""

import importlib


def test_render_function_exists():
    """pages/ore.py must expose a render() function."""
    import pages.ore as ore_module

    assert hasattr(ore_module, "render")
    assert callable(ore_module.render)


def test_no_services_utils_import():
    """pages/ore.py must NOT import from services.utils (deprecated)."""
    import ast
    import pathlib

    source = pathlib.Path("pages/ore.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert (
                "services.utils" not in node.module
            ), f"Found forbidden import from services.utils in pages/ore.py"


def test_imports_from_enterprise_vision_ai():
    """pages/ore.py must import from enterprise_vision_ai.utils."""
    import ast
    import pathlib

    source = pathlib.Path("pages/ore.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "enterprise_vision_ai" in node.module:
                found = True
                break
    assert found, "pages/ore.py must import from enterprise_vision_ai.utils"
