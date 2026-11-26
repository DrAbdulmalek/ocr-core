"""Optional CPU-first adapter for Datalab's Indra OCR engine.

Indra runs through ONNX Runtime and is imported only when this adapter is used.
No confidence is fabricated: confidence stays 0.0 when the engine provides no
usable per-line confidence values.
"""
from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import Optional

from ocr_core.engines.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)


class IndraEngine(OCREngine):
    """Datalab Indra OCR adapter for images and PDFs."""

    name = "indra"

    def __init__(self, threads: int = 4, math: bool = False) -> None:
        if threads < 1:
            raise ValueError("threads must be >= 1")
        self.threads = threads
        self.math = math

    def available(self) -> bool:
        """Return whether the optional indra-ocr package is importable."""
        try:
            return importlib.util.find_spec("indra_ocr") is not None
        except (ImportError, ValueError):
            return False

    def process_image(self, image_path) -> OCRResult:
        return self._process_document(image_path, max_pages=1)

    def process_pdf(self, pdf_path, max_pages: Optional[int] = None) -> OCRResult:
        return self._process_document(pdf_path, max_pages=max_pages)

    def _process_document(self, document_path, max_pages: Optional[int]) -> OCRResult:
        path = Path(document_path)
        if not path.is_file():
            return OCRResult(
                text="",
                engine=self.name,
                error=f"Input file not found or not a file: {path}",
                pages=0,
            )

        started = time.perf_counter()
        try:
            import indra_ocr
        except ImportError as exc:
            return OCRResult(
                text="",
                engine=self.name,
                error="Indra OCR is optional; install with "
                "'pip install marathon-ocr-core[indra]'. "
                f"Import detail: {exc}",
                pages=0,
            )

        try:
            converter = indra_ocr.load(threads=self.threads, math=self.math)
            page_records: list[dict] = []
            texts: list[str] = []
            confidences: list[float] = []

            for page_number, page in converter.convert(str(path)):
                if max_pages is not None and len(page_records) >= max_pages:
                    break
                text = getattr(page, "text", None) or getattr(page, "markdown", None) or ""
                texts.append(str(text))
                lines = []
                for line in getattr(page, "lines", []) or []:
                    line_text = getattr(line, "text", "")
                    line_conf = getattr(line, "conf", None)
                    line_box = getattr(line, "box", None)
                    record = {"text": str(line_text)}
                    if line_box is not None:
                        record["box"] = line_box
                    if isinstance(line_conf, (int, float)) and 0.0 <= line_conf <= 1.0:
                        record["confidence"] = float(line_conf)
                        confidences.append(float(line_conf))
                    lines.append(record)

                page_records.append({
                    "page": page_number,
                    "text": str(text),
                    "markdown": getattr(page, "markdown", None),
                    "rotation": getattr(page, "rotation", None),
                    "skew": getattr(page, "skew", None),
                    "source": getattr(page, "source", None),
                    "lines": lines,
                    "blocks": getattr(page, "blocks", []) or [],
                })

            elapsed = time.perf_counter() - started
            return OCRResult(
                text="\n\n".join(texts),
                engine=self.name,
                confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
                processing_time=elapsed,
                pages=len(page_records),
                meta={
                    "provider": "datalab-to/indra",
                    "runtime": "onnxruntime",
                    "threads": self.threads,
                    "math": self.math,
                    "page_results": page_records,
                },
            )
        except Exception as exc:
            logger.exception("Indra OCR failed for %s", path)
            return OCRResult(
                text="",
                engine=self.name,
                processing_time=time.perf_counter() - started,
                pages=0,
                error=f"{type(exc).__name__}: {exc}",
                meta={"provider": "datalab-to/indra", "runtime": "onnxruntime"},
            )
