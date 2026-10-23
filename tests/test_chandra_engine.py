from pathlib import Path
import subprocess

from ocr_core.engines.chandra import ChandraEngine


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
