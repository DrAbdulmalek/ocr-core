"""Optional adapter for a locally installed, licensed ABBYY FineReader bridge.

This module does not bundle ABBYY software, models, weights, APK contents, or
reverse-engineered binaries. The caller must provide a command template that
invokes an ABBYY product/SDK they are authorized to use, or a local wrapper
around that licensed installation.

Template arguments are passed directly to subprocess (never through a shell).
Supported placeholders:
  {input}  absolute path to the input image
  {output} temporary path for a text output file (file mode only)

No confidence score is inferred: confidence remains 0.0 unless a future,
validated adapter can parse a genuine score from the installed product.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional, Sequence

from ocr_core.engines.base import OCRResult, OCREngine


class AbbyyFineReaderEngine(OCREngine):
    """Run an externally installed ABBYY-compatible command/SDK bridge.

    Args:
        command: Argument vector with {input}; in output_mode="file", include
            {output}. Use the exact command syntax documented by your licensed
            local installation or a wrapper script you control.
        output_mode: "file" reads text from the temporary {output} path;
            "stdout" reads UTF-8 text emitted to stdout.
        timeout: Maximum seconds allowed for a single OCR invocation.
        encoding: Text encoding used when reading a file-mode result.
    """

    name = "abbyy-finereader"

    def __init__(
        self,
        command: Optional[Sequence[str]] = None,
        *,
        output_mode: Literal["file", "stdout"] = "file",
        timeout: float = 120.0,
        encoding: str = "utf-8",
    ) -> None:
        self.command = tuple(str(part) for part in command) if command else ()
        self.output_mode = output_mode
        self.timeout = float(timeout)
        self.encoding = encoding
        if output_mode not in {"file", "stdout"}:
            raise ValueError("output_mode must be 'file' or 'stdout'")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.command and not any("{input}" in part for part in self.command):
            raise ValueError("command must contain the {input} placeholder")
        if self.command and output_mode == "file" and not any(
            "{output}" in part for part in self.command
        ):
            raise ValueError("file output mode requires the {output} placeholder")

    def available(self) -> bool:
        """Return true only when a configured executable resolves on this host."""
        if not self.command:
            return False
        executable = self.command[0]
        return shutil.which(executable) is not None or (
            Path(executable).is_file() and Path(executable).exists()
        )

    def process_image(self, image_path) -> OCRResult:
        started = time.perf_counter()
        path = Path(image_path).expanduser()
        if not path.is_file():
            return OCRResult(
                engine=self.name,
                error=f"file not found: {path}",
                processing_time=time.perf_counter() - started,
            )
        if not self.command:
            return OCRResult(
                engine=self.name,
                error="ABBYY bridge is not configured; supply a licensed local command template",
                processing_time=time.perf_counter() - started,
            )
        if not self.available():
            return OCRResult(
                engine=self.name,
                error=f"configured executable not found: {self.command[0]}",
                processing_time=time.perf_counter() - started,
            )

        try:
            with tempfile.TemporaryDirectory(prefix="ocr-core-abbyy-") as temp_dir:
                output_path = Path(temp_dir) / "result.txt"
                replacements = {
                    "{input}": str(path.resolve()),
                    "{output}": str(output_path),
                }
                args = []
                for item in self.command:
                    rendered = item
                    for placeholder, value in replacements.items():
                        rendered = rendered.replace(placeholder, value)
                    args.append(rendered)

                completed = subprocess.run(
                    args,
                    shell=False,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    detail = detail[-1000:] if detail else "no diagnostic output"
                    return OCRResult(
                        engine=self.name,
                        error=f"ABBYY bridge exited with code {completed.returncode}: {detail}",
                        processing_time=time.perf_counter() - started,
                        meta={"returncode": completed.returncode, "output_mode": self.output_mode},
                    )

                if self.output_mode == "stdout":
                    text = completed.stdout
                else:
                    if not output_path.is_file():
                        return OCRResult(
                            engine=self.name,
                            error="ABBYY bridge succeeded but did not create the requested output file",
                            processing_time=time.perf_counter() - started,
                            meta={"returncode": completed.returncode, "output_mode": self.output_mode},
                        )
                    text = output_path.read_text(encoding=self.encoding, errors="replace")

                return OCRResult(
                    text=text.strip(),
                    engine=self.name,
                    confidence=0.0,
                    processing_time=time.perf_counter() - started,
                    meta={
                        "returncode": completed.returncode,
                        "output_mode": self.output_mode,
                        "executable": Path(self.command[0]).name,
                        "confidence_status": "unknown_not_provided_by_adapter",
                    },
                )
        except subprocess.TimeoutExpired:
            return OCRResult(
                engine=self.name,
                error=f"ABBYY bridge timed out after {self.timeout:g} seconds",
                processing_time=time.perf_counter() - started,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return OCRResult(
                engine=self.name,
                error=f"{type(exc).__name__}: {exc}",
                processing_time=time.perf_counter() - started,
            )

    def process_pdf(self, pdf_path, max_pages: Optional[int] = None) -> OCRResult:
        return OCRResult(
            engine=self.name,
            error="PDF processing is not implemented by this adapter; convert pages or use a documented PDF-capable bridge",
            meta={"max_pages_requested": max_pages},
        )
