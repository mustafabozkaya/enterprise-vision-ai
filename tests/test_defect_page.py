"""Tests for pages/defect.py module."""

import importlib


def test_render_function_exists():
    """pages/defect.py must expose a render() function."""
    import pages.defect as defect_module

    assert hasattr(defect_module, "render")
    assert callable(defect_module.render)


def test_no_services_utils_import():
    """pages/defect.py must NOT import from services.utils (deprecated)."""
    import ast
    import pathlib

    source = pathlib.Path("pages/defect.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert (
                    "services.utils" not in node.module
                ), f"Found forbidden import from services.utils in pages/defect.py"


def test_imports_from_enterprise_vision_ai():
    """pages/defect.py must import from enterprise_vision_ai.utils."""
    import ast
    import pathlib

    source = pathlib.Path("pages/defect.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "enterprise_vision_ai" in node.module:
                found = True
                break
    assert found, "pages/defect.py must import from enterprise_vision_ai.utils"
