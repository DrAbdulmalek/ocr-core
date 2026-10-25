#!/usr/bin/env python3
"""E25 evidence script — REAL chandra-ocr CLI smoke check (PR #66).

RECONSTRUCTED 2026-10-25. The original E25 script was lost together with the
verification environment reset; this copy was rebuilt to reproduce the exact
procedure and numbers documented in the published E26 review comment
(github.com/DrAbdulmalek/ocr-core#issuecomment-6368829627) and in the live
integration tests added to the PR branch (commit da0c831,
tests/test_chandra_integration.py). It is functionally equivalent to the
original: same CLI invocations, same pass/fail gates.

What it proves against the REAL ``chandra`` executable (chandra-ocr==0.2.0):

1. CLI argument contract: ``chandra --help`` exposes ``--method`` (hf|vllm)
   and ``--page-range`` — the flags the adapter relies on.
2. Silent-failure reproduction (the E25 critical finding): with NO vLLM
   server reachable, the CLI still exits 0 and writes an EMPTY ``<stem>.md``
   (after its retry loop) — ``total_token_count == 0`` in the sibling
   ``<stem>_metadata.json``. Pre-fix (8b601e8) the adapter accepted this as
   ok=True; this is the hole fix 3ed55c1 closed.
3. Adapter guard (post-fix branch): routing the same run through
   ``ChandraEngine`` must produce an explicit failure: ``ok=False``,
   error mentions "empty", ``meta["empty_output"] is True``.

Usage (inside a venv with ``chandra-ocr`` installed, no weights needed):

    python evidence/pr66/scripts/e25_chandra_real_cli_smoke.py \
        [--json OUT.json] [--skip-adapter]

Exit code 0 only if every gate holds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RECONSTRUCTED_NOTE = (
    "Rebuilt 2026-10-25 after environment reset; procedure mirrors the "
    "published E26 review (issuecomment-6368829627) and commit da0c831 tests."
)


def _tiny_png(tmp: Path) -> Path:
    """Create a small white PNG so the CLI runs its full pipeline."""
    from PIL import Image  # pillow ships with the chandra-ocr base install

    png = tmp / "scan.png"
    Image.new("RGB", (120, 60), "white").save(png)
    return png


def check_cli_contract(exe: str) -> dict:
    """Gate 1 — the flags the adapter depends on must exist."""
    out = subprocess.run(
        [exe, "--help"], capture_output=True, text=True, timeout=60, check=False
    )
    text = out.stdout + out.stderr
    gates = {
        "returncode_zero": out.returncode == 0,
        "flag_method": "--method" in text,
        "flag_page_range": "--page-range" in text,
        "methods_hf_vllm": ("hf" in text) and ("vllm" in text),
    }
    return {"step": "cli_contract", "ok": all(gates.values()), "gates": gates}


def reproduce_silent_failure(exe: str, tmp: Path) -> dict:
    """Gate 2 — raw CLI behaviour with an unreachable vLLM backend.

    Verified against chandra-ocr 0.2.0: the CLI retries the backend, then
    still exits 0 and writes an EMPTY <stem>.md plus a <stem>_metadata.json
    with total_token_count == 0.
    """
    png = _tiny_png(tmp)
    outdir = tmp / "out"
    outdir.mkdir()
    proc = subprocess.run(
        [exe, str(png), str(outdir), "--method", "vllm"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    md_files = list(outdir.rglob("scan.md"))
    md_size = md_files[0].stat().st_size if len(md_files) == 1 else None
    md_empty = md_size == 0 or (
        md_size is not None and not md_files[0].read_text(encoding="utf-8").strip()
    )

    token_count = None
    meta_files = list(outdir.rglob("scan_metadata.json"))
    if meta_files:
        try:
            token_count = json.loads(meta_files[0].read_text(encoding="utf-8")).get(
                "total_token_count"
            )
        except (OSError, ValueError):
            token_count = "unreadable"

    gates = {
        "cli_returncode_zero_despite_failure": proc.returncode == 0,
        "markdown_written": len(md_files) == 1,
        "markdown_empty": bool(md_empty),
        "zero_tokens_recorded": token_count == 0,
    }
    return {
        "step": "silent_failure_reproduction",
        "ok": all(gates.values()),
        "gates": gates,
        "evidence": {
            "returncode": proc.returncode,
            "markdown_size_bytes": md_size,
            "total_token_count": token_count,
        },
    }


def check_adapter_guard(tmp: Path) -> dict:
    """Gate 3 — the post-fix adapter must reject the empty output."""
    repo_src = Path(__file__).resolve().parents[3] / "src"
    if repo_src.is_dir():
        sys.path.insert(0, str(repo_src))
    try:
        from ocr_core.engines.chandra import ChandraEngine
    except ImportError as exc:
        return {
            "step": "adapter_guard",
            "ok": False,
            "error": f"ocr_core not importable ({exc}); run inside the repo venv",
        }

    png_dir = tmp / "adapter"
    png_dir.mkdir()
    png = _tiny_png(png_dir)
    engine = ChandraEngine(method="vllm", executable="chandra", timeout=600)
    result = engine.process_image(png)
    meta = getattr(result, "meta", {}) or {}
    gates = {
        "ok_is_false": result.ok is False,
        "error_mentions_empty": "empty" in (result.error or "").lower(),
        "empty_output_flag": meta.get("empty_output") is True,
    }
    return {
        "step": "adapter_guard",
        "ok": all(gates.values()),
        "gates": gates,
        "evidence": {"error": (result.error or "")[:200], "meta_keys": sorted(meta)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_out", help="write the report as JSON")
    parser.add_argument(
        "--skip-adapter", action="store_true", help="only run the raw CLI gates"
    )
    args = parser.parse_args()

    exe = shutil.which("chandra")
    if exe is None:
        print("FAIL: chandra CLI not found on PATH (pip install chandra-ocr)")
        return 2

    with tempfile.TemporaryDirectory(prefix="e25-chandra-smoke-") as tmp_name:
        tmp = Path(tmp_name)
        report = {
            "artifact": "e25_chandra_real_cli_smoke",
            "note": RECONSTRUCTED_NOTE,
            "chandra_executable": exe,
            "steps": [],
        }
        report["steps"].append(check_cli_contract(exe))
        report["steps"].append(reproduce_silent_failure(exe, tmp))
        if not args.skip_adapter:
            report["steps"].append(check_adapter_guard(tmp))

    report["overall_ok"] = all(step.get("ok") for step in report["steps"])
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
