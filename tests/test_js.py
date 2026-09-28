"""Run the pure-JS unit tests (tests/js) with node's built-in runner."""

import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS = sorted((Path(__file__).parent / "js").glob("*.test.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("path", JS_TESTS, ids=lambda p: p.name)
def test_node(path):
    r = subprocess.run(["node", "--test", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
