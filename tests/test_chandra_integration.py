"""Integration tests: the REAL chandra-ocr CLI against the adapter.

Unlike tests/test_chandra_engine.py (mocked unit tests), these run the actual
`chandra` executable (chandra-ocr, base install) and verify the CLI contract
and failure semantics that mocks cannot prove. They download NO model weights
and are skipped unless CHANDRA_INTEGRATION=1 is set, so CI stays hermetic.

Local run (with chandra-ocr installed in a venv):

    CHANDRA_INTEGRATION=1 PATH="<venv>/bin:$PATH" \
        pytest tests/test_chandra_integration.py -v

Verified contract (chandra-ocr==0.2.0, see docs/chandra_ocr.md):
- `chandra <input> <output> --method [hf|vllm]` positional CLI
- unreachable vLLM backend: CLI retries, then exits 0 with an EMPTY .md
  (the silent-failure hole closed by fix 3ed55c1; adapter must reject it)
- hf backend without torch: non-zero exit, surfaced as an explicit error
"""
import os
import shutil
import subprocess

import pytest

from ocr_core.engines.chandra import ChandraEngine

pytestmark = pytest.mark.skipif(
    os.environ.get("CHANDRA_INTEGRATION") != "1",
    reason="integration tests need the real chandra CLI (CHANDRA_INTEGRATION=1)",
)


def _require_cli() -> str:
    exe = shutil.which("chandra")
    if exe is None:
        pytest.skip("chandra CLI not found on PATH")
    return exe


def _tiny_png(tmp_path):
    pytest.importorskip("PIL.Image")  # pillow ships with chandra-ocr base
    from PIL import Image

    png = tmp_path / "scan.png"
    Image.new("RGB", (120, 60), "white").save(png)
    return png


def test_real_cli_argument_contract():
    """`chandra --help` must expose the flags the adapter relies on."""
    exe = _require_cli()
    out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60)
    text = out.stdout + out.stderr
    assert "--method" in text
    assert "--page-range" in text
    assert "hf" in text and "vllm" in text


def test_real_vllm_without_server_is_explicit_failure(tmp_path):
    """Reproduces the E25 silent failure with the real CLI, end to end.

    chandra-ocr 0.2.0 retries the unreachable vLLM backend, then exits 0 and
    writes an EMPTY <stem>.md. Before fix 3ed55c1 the adapter returned
    ok=True with empty text; it must now report an explicit failure.
    """
    exe = _require_cli()
    png = _tiny_png(tmp_path)

    engine = ChandraEngine(method="vllm", executable=exe, timeout=300)
    result = engine.process_image(png)

    assert not result.ok
    assert "empty" in (result.error or "").lower()
    assert result.meta.get("empty_output") is True


def test_real_hf_backend_crash_is_surfaced(tmp_path):
    """A real hf-backend crash (e.g. torch missing) must never look like
    success. Skipped when torch IS importable, because then the CLI would
    attempt genuine model loading instead of crashing."""
    exe = _require_cli()
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch importable; hf path would attempt real inference")

    png = _tiny_png(tmp_path)
    engine = ChandraEngine(method="hf", executable=exe, timeout=300)
    result = engine.process_image(png)

    assert not result.ok
    assert "exited with status" in (result.error or "")
