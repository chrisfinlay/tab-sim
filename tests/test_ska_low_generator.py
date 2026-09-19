"""The data-maintenance tool rejects unverified inputs without executing code."""
import ast
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "generate_ska_low", Path(__file__).resolve().parents[1] / "tools/generate_ska_low.py"
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_rejects_changed_source(tmp_path):
    source = tmp_path / "layout.json"
    source.write_text('{"receptors": []}')
    with pytest.raises(ValueError, match="not the pinned layout source"):
        generator.read_verified(source, "layout")


def test_membership_parser_never_executes_python():
    expr = ast.parse('"C1," + "S8-1"', mode="eval").body
    assert generator.literal_string(expr) == "C1,S8-1"
    with pytest.raises(ValueError, match="literal station-list"):
        generator.literal_string(ast.parse('__import__("os").getcwd()', mode="eval").body)
