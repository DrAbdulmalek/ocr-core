from __future__ import annotations

import sys
from pathlib import Path

from ocr_core.engines.abbyy import AbbyyFineReaderEngine


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "scan.png"
    path.write_bytes(b"test-image-placeholder")
    return path


def test_file_mode_reads_output_and_does_not_invent_confidence(tmp_path):
    image = _image(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[2]).write_text('تقرير طبي', encoding='utf-8')",
        "{input}",
        "{output}",
    ]
    engine = AbbyyFineReaderEngine(command)

    result = engine.process_image(image)

    assert result.ok
    assert result.text == "تقرير طبي"
    assert result.engine == "abbyy-finereader"
    assert result.confidence == 0.0
    assert result.meta["confidence_status"] == "unknown_not_provided_by_adapter"


def test_stdout_mode_reads_stdout(tmp_path):
    image = _image(tmp_path)
    command = [sys.executable, "-c", "print('recognized text')", "{input}"]
    engine = AbbyyFineReaderEngine(command, output_mode="stdout")

    result = engine.process_image(image)

    assert result.ok
    assert result.text == "recognized text"
    assert result.confidence == 0.0


def test_unconfigured_engine_is_not_available_and_fails_visibly(tmp_path):
    engine = AbbyyFineReaderEngine()
    result = engine.process_image(_image(tmp_path))

    assert not engine.available()
    assert result.error
    assert "not configured" in result.error


def test_missing_executable_fails_visibly(tmp_path):
    engine = AbbyyFineReaderEngine(
        ["/definitely/not/an/installed/abbyy-bridge", "{input}", "{output}"]
    )

    result = engine.process_image(_image(tmp_path))

    assert not engine.available()
    assert result.error
    assert "executable not found" in result.error


def test_nonzero_exit_is_returned_as_error(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import sys; print('bridge failed', file=sys.stderr); sys.exit(7)",
        "{input}",
        "{output}",
    ]
    result = AbbyyFineReaderEngine(command).process_image(_image(tmp_path))

    assert result.error
    assert "code 7" in result.error
    assert "bridge failed" in result.error


def test_timeout_is_returned_as_error(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(2)", "{input}"]
    result = AbbyyFineReaderEngine(
        command, output_mode="stdout", timeout=0.05
    ).process_image(_image(tmp_path))

    assert result.error
    assert "timed out" in result.error


def test_missing_input_fails_before_invoking_bridge(tmp_path):
    engine = AbbyyFineReaderEngine(
        [sys.executable, "-c", "print('should not run')", "{input}"],
        output_mode="stdout",
    )

    result = engine.process_image(tmp_path / "missing.png")

    assert result.error
    assert "file not found" in result.error


def test_file_mode_requires_output_placeholder():
    try:
        AbbyyFineReaderEngine([sys.executable, "-c", "print('x')", "{input}"])
    except ValueError as exc:
        assert "output" in str(exc)
    else:
        raise AssertionError("file mode must require an {output} placeholder")


def test_from_env_without_command_is_unconfigured(monkeypatch):
    monkeypatch.delenv("OCR_CORE_ABBYY_COMMAND", raising=False)
    engine = AbbyyFineReaderEngine.from_env()
    assert not engine.available()


def test_from_env_parses_json_argument_vector(monkeypatch):
    monkeypatch.setenv(
        "OCR_CORE_ABBYY_COMMAND",
        '["python", "-c", "print(1)", "{input}"]',
    )
    monkeypatch.setenv("OCR_CORE_ABBYY_OUTPUT_MODE", "stdout")
    monkeypatch.setenv("OCR_CORE_ABBYY_TIMEOUT", "17")
    engine = AbbyyFineReaderEngine.from_env()
    assert engine.command == ("python", "-c", "print(1)", "{input}")
    assert engine.output_mode == "stdout"
    assert engine.timeout == 17


def test_from_env_rejects_shell_string(monkeypatch):
    monkeypatch.setenv("OCR_CORE_ABBYY_COMMAND", "wine bridge.exe --input {input}")
    try:
        AbbyyFineReaderEngine.from_env()
    except ValueError as exc:
        assert "JSON array" in str(exc)
    else:
        raise AssertionError("a shell command string must be rejected")


def test_from_env_rejects_non_string_array_items(monkeypatch):
    monkeypatch.setenv("OCR_CORE_ABBYY_COMMAND", '["bridge", 3, "{input}", "{output}"]')
    try:
        AbbyyFineReaderEngine.from_env()
    except ValueError as exc:
        assert "strings" in str(exc)
    else:
        raise AssertionError("non-string command arguments must be rejected")
