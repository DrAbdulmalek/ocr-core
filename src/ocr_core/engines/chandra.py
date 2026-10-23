"""Optional adapter for Datalab Chandra OCR via its documented CLI.

The adapter invokes the local CLI without a shell and does not call a hosted API.
Model weights are not bundled. Confidence remains unknown unless a future,
verified Chandra output contract supplies a genuine confidence value.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from ocr_core.engines.base import OCREngine, OCRResult


class ChandraEngine(OCREngine):
    """Run Chandra OCR using a local Hugging Face or configured vLLM backend."""

    name = "chandra"

    def __init__(
        self,
        method: str = "hf",
        executable: str = "chandra",
        timeout: int = 900,
    ) -> None:
        if method not in {"hf", "vllm"}:
            raise ValueError("method must be 'hf' or 'vllm'")
        if timeout < 1:
            raise ValueError("timeout must be >= 1")
        self.method = method
        self.executable = executable
        self.timeout = timeout

    def available(self) -> bool:
        """Return whether the configured Chandra CLI executable is on PATH."""
        return shutil.which(self.executable) is not None

    def process_image(self, image_path) -> OCRResult:
        return self._process_document(image_path, max_pages=None, is_image=True)

    def process_pdf(self, pdf_path, max_pages: Optional[int] = None) -> OCRResult:
        if max_pages is not None and max_pages < 1:
            return OCRResult(
                text="", engine=self.name,
                error="max_pages must be >= 1 when provided", pages=0,
            )
        return self._process_document(pdf_path, max_pages=max_pages, is_image=False)

    def _process_document(self, document_path, max_pages, is_image: bool) -> OCRResult:
        path = Path(document_path)
        if not path.is_file():
            return OCRResult(
                text="", engine=self.name,
                error=f"Input file not found or not a file: {path}", pages=0,
            )
        if not self.available():
            return OCRResult(
                text="", engine=self.name,
                error=(
                    f"Chandra CLI '{self.executable}' is unavailable; install "
                    "'marathon-ocr-core[chandra]' and the selected Chandra backend."
                ),
                pages=0,
                meta={"provider": "datalab-to/chandra", "method": self.method},
            )

        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="ocr-core-chandra-") as tmp:
                output_dir = Path(tmp) / "output"
                output_dir.mkdir()
                command = [
                    self.executable, str(path), str(output_dir),
                    "--method", self.method,
                ]
                if not is_image and max_pages is not None:
                    command.extend(["--page-range", f"1-{max_pages}"])

                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    shell=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    return OCRResult(
                        text="", engine=self.name,
                        processing_time=time.perf_counter() - started,
                        pages=0,
                        error=(
                            f"Chandra CLI exited with status {completed.returncode}"
                            + (f": {detail[-2000:]}" if detail else "")
                        ),
                        meta={
                            "provider": "datalab-to/chandra",
                            "method": self.method,
                            "returncode": completed.returncode,
                        },
                    )

                expected = list(output_dir.rglob(f"{path.stem}.md"))
                if not expected:
                    markdown_files = list(output_dir.rglob("*.md"))
                    if len(markdown_files) == 1:
                        expected = markdown_files
                if len(expected) != 1:
                    return OCRResult(
                        text="", engine=self.name,
                        processing_time=time.perf_counter() - started,
                        pages=0,
                        error=(
                            "Chandra completed but did not produce one unambiguous "
                            "Markdown result file."
                        ),
                        meta={
                            "provider": "datalab-to/chandra",
                            "method": self.method,
                            "markdown_candidates": len(expected),
                        },
                    )

                text = expected[0].read_text(encoding="utf-8")
                return OCRResult(
                    text=text,
                    engine=self.name,
                    confidence=0.0,
                    processing_time=time.perf_counter() - started,
                    # The CLI's metadata schema is not yet parsed, so PDF page
                    # counts remain unknown rather than reporting a requested cap.
                    pages=(1 if is_image else 0),
                    meta={
                        "provider": "datalab-to/chandra",
                        "method": self.method,
                        "runtime": "chandra-cli",
                        "output_format": "markdown",
                        "confidence_status": "unknown",
                        "page_count_status": "known_image" if is_image else "unknown",
                        "requested_max_pages": max_pages,
                    },
                )
        except subprocess.TimeoutExpired:
            return OCRResult(
                text="", engine=self.name,
                processing_time=time.perf_counter() - started,
                pages=0,
                error=f"Chandra CLI timed out after {self.timeout} seconds",
                meta={"provider": "datalab-to/chandra", "method": self.method},
            )
        except (OSError, UnicodeError) as exc:
            return OCRResult(
                text="", engine=self.name,
                processing_time=time.perf_counter() - started,
                pages=0,
                error=f"{type(exc).__name__}: {exc}",
                meta={"provider": "datalab-to/chandra", "method": self.method},
            )
