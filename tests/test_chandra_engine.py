from pathlib import Path
import json
import subprocess

from ocr_core.engines.chandra import ChandraEngine


def _fake_cli_result(command, md_text, metadata=None):
    """Simulate a real chandra-ocr 0.2.0 CLI run inside the adapter's tempdir.

    Verified real output layout (chandra/scripts/cli.py::save_merged_output,
    confirmed by a live run against chandra-ocr==0.2.0):
      <output>/<stem>/<stem>.md and sibling <stem>/<stem>_metadata.json
    The real CLI exits 0 even when inference fails (e.g. unreachable vLLM
    backend) and then writes an EMPTY .md plus metadata with total_token_count 0.
    """
    output_dir = Path(command[2])
    stem = Path(command[1]).stem
    result_dir = output_dir / stem
    result_dir.mkdir(parents=True)
    (result_dir / f"{stem}.md").write_text(md_text, encoding="utf-8")
    if metadata is not None:
        (result_dir / f"{stem}_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_chandra_runs_cli_and_reads_markdown(monkeypatch, tmp_path):
    source = tmp_path / "scan.png"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        assert command[command.index("--method") + 1] == "hf"
        output_dir = Path(command[2])
        result_dir = output_dir / source.stem
        result_dir.mkdir(parents=True)
        (result_dir / f"{source.stem}.md").write_text("تقرير طبي", encoding="utf-8")
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine(method="hf").process_image(source)

    assert result.ok
    assert result.text == "تقرير طبي"
    assert result.engine == "chandra"
    assert result.confidence == 0.0
    assert result.pages == 1
    assert result.meta["confidence_status"] == "unknown"


def test_chandra_missing_input(tmp_path):
    result = ChandraEngine().process_image(tmp_path / "missing.png")
    assert not result.ok
    assert result.pages == 0


def test_chandra_missing_cli(monkeypatch, tmp_path):
    source = tmp_path / "scan.png"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: None)

    result = ChandraEngine().process_image(source)

    assert not result.ok
    assert "unavailable" in result.error


def test_chandra_rejects_invalid_method():
    try:
        ChandraEngine(method="cloud")
    except ValueError as exc:
        assert "method" in str(exc)
    else:
        raise AssertionError("invalid method should be rejected")


def test_chandra_pdf_page_limit_is_passed_to_cli(monkeypatch, tmp_path):
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        out = Path(command[2]) / source.stem
        out.mkdir(parents=True)
        (out / f"{source.stem}.md").write_text("page text", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_pdf(source, max_pages=2)

    assert result.ok
    assert seen["command"][seen["command"].index("--page-range") + 1] == "1-2"
    assert result.pages == 0
    assert result.meta["page_count_status"] == "unknown"


def test_chandra_timeout_is_reported(monkeypatch, tmp_path):
    source = tmp_path / "scan.png"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", timeout)
    result = ChandraEngine(timeout=1).process_image(source)

    assert not result.ok
    assert "timed out" in result.error


# --- E26: silent-failure guards (reproduces the defect found by live testing) ---

def test_chandra_empty_markdown_with_zero_exit_is_failure(monkeypatch, tmp_path):
    """Reproduces the E25 finding: chandra-ocr 0.2.0 exits 0 and writes an
    empty .md when inference fails (vLLM backend unreachable). The adapter
    must NOT report this as a successful blank result."""
    source = tmp_path / "scan.png"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        return _fake_cli_result(
            command, md_text="",
            metadata={"num_pages": 1, "total_token_count": 0},
        )

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_image(source)

    assert not result.ok
    assert "empty" in (result.error or "").lower()
    assert result.text == ""
    assert result.meta["empty_output"] is True
    assert result.meta["generation_evidence"] == "zero_tokens_recorded"


def test_chandra_whitespace_markdown_is_failure(monkeypatch, tmp_path):
    source = tmp_path / "scan.png"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        return _fake_cli_result(command, md_text="  \n\t  \n")

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_image(source)

    assert not result.ok
    assert "empty" in (result.error or "").lower()


def test_chandra_missing_markdown_is_failure(monkeypatch, tmp_path):
    source = tmp_path / "scan.png"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_image(source)

    assert not result.ok
    assert "unambiguous" in (result.error or "")


def test_chandra_metadata_num_pages_is_recorded(monkeypatch, tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        return _fake_cli_result(
            command, md_text="page one\n\npage two",
            metadata={"num_pages": 3, "total_token_count": 4213},
        )

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_pdf(source)

    assert result.ok
    assert result.pages == 3
    assert result.meta["page_count_status"] == "known_metadata"
    assert result.meta["metadata_status"] == "parsed"


def test_chandra_missing_metadata_is_defensive(monkeypatch, tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        return _fake_cli_result(command, md_text="real text")

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_pdf(source)

    assert result.ok
    assert result.pages == 0
    assert result.meta["page_count_status"] == "unknown"
    assert result.meta["metadata_status"] == "missing"


def test_chandra_corrupt_metadata_is_defensive(monkeypatch, tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        output_dir = Path(command[2])
        stem = Path(command[1]).stem
        result_dir = output_dir / stem
        result_dir.mkdir(parents=True)
        (result_dir / f"{stem}.md").write_text("real text", encoding="utf-8")
        (result_dir / f"{stem}_metadata.json").write_text("{not-json", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_pdf(source)

    assert result.ok
    assert result.pages == 0
    assert result.meta["metadata_status"] == "corrupt"


def test_chandra_zero_token_metadata_is_flagged_not_fatal(monkeypatch, tmp_path):
    """The proven 0.2.0 contract is: failure -> empty md AND zero tokens.
    The converse (non-empty md with zero recorded tokens) is NOT proven as a
    failure, so it is surfaced as evidence without rejecting the result."""
    source = tmp_path / "scan.png"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        return _fake_cli_result(
            command, md_text="some text",
            metadata={"num_pages": 1, "total_token_count": 0},
        )

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_image(source)

    assert result.ok
    assert result.meta["generation_evidence"] == "zero_tokens_recorded"


def test_chandra_nonzero_exit_is_failure(monkeypatch, tmp_path):
    """Unit-level mirror of the live E25 test: a real hf-backend crash
    (ModuleNotFoundError: torch) surfaces as an explicit error, never success."""
    source = tmp_path / "scan.png"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr("ocr_core.engines.chandra.shutil.which", lambda _: "/usr/bin/chandra")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="ModuleNotFoundError: No module named 'torch'"
        )

    monkeypatch.setattr("ocr_core.engines.chandra.subprocess.run", fake_run)
    result = ChandraEngine().process_image(source)

    assert not result.ok
    assert "exited with status 1" in (result.error or "")
