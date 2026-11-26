"""CLI contract tests for the local ModelStore smoke-test helper."""
import subprocess
import sys
from pathlib import Path


def test_model_store_smoke_cli_help_is_dependency_light():
    script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_model_store.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--store" in result.stdout
    assert "--trust-remote-code" in result.stdout
