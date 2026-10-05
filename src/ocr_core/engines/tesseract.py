"""Tesseract engine (optional extra: pip install .[tesseract]).

Needs the ``tesseract`` binary on PATH plus pytesseract + Pillow.
PDF support additionally uses pymupdf to rasterize pages.
Engine-level failures are reported via ``OCRResult.error`` — never raised.

Arabic + English by default (``lang="ara+eng"``). Bundled tessdata models
(Apache-2.0, see ``ocr_core/resources/tessdata_fast/PROVENANCE.md``) are
auto-discovered and passed via ``--tessdata-dir`` — no global install needed.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from ocr_core.engines.base import OCRResult, OCREngine

_BUNDLED_TESSDATA = (
    Path(__file__).resolve().parent.parent / "resources" / "tessdata_fast"
)


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(
        self,
        lang: str = "ara+eng",
        psm: str = "6",
        dpi: int = 200,
        tessdata_dir: Optional[str] = None,
    ):
        self.lang = lang
        self.psm = psm
        self.dpi = dpi
        self.tessdata_dir = tessdata_dir or self._bundled_tessdata_dir()

    @staticmethod
    def _bundled_tessdata_dir() -> Optional[str]:
        """Bundled Apache-2.0 models (ara+eng), only if they actually exist."""
        try:
            if (_BUNDLED_TESSDATA / "ara.traineddata").is_file() and (
                _BUNDLED_TESSDATA / "eng.traineddata"
            ).is_file():
                return str(_BUNDLED_TESSDATA)
        except OSError:
            pass
        return None

    def _config(self) -> str:
        cfg = f"--psm {self.psm}"
        if self.tessdata_dir:
            cfg += f" --tessdata-dir {self.tessdata_dir}"
        return cfg

    def available(self) -> bool:
        return shutil.which("tesseract") is not None

    def process_image(self, image_path) -> OCRResult:
        start = time.perf_counter()
        path = str(image_path)
        if not os.path.isfile(path):
            return OCRResult(engine=self.name, error=f"file not found: {path}")
        if not self.available():
            return OCRResult(engine=self.name, error="tesseract binary not on PATH")
        try:
            import pytesseract
        except ImportError:
            return OCRResult(
                engine=self.name,
                error="pytesseract not installed (pip install .[tesseract])",
            )
        try:
            from PIL import Image

            text = pytesseract.image_to_string(
                Image.open(path), lang=self.lang, config=self._config()
            )
        except Exception as exc:  # engine failure -> reported, never raised
            return OCRResult(engine=self.name, error=f"{type(exc).__name__}: {exc}")
        return OCRResult(
            text=text.strip(),
            engine=self.name,
            confidence=0.0,  # never invented
            processing_time=time.perf_counter() - start,
            meta={
                "lang": self.lang,
                "psm": self.psm,
                "tessdata_dir": self.tessdata_dir,
            },
        )

    def process_pdf(self, pdf_path, max_pages: Optional[int] = None) -> OCRResult:
        start = time.perf_counter()
        path = str(pdf_path)
        if not os.path.isfile(path):
            return OCRResult(engine=self.name, error=f"file not found: {path}")
        try:
            import fitz  # pymupdf
        except ImportError:
            return OCRResult(engine=self.name, error="pymupdf not installed (pip install pymupdf)")
        try:
            from PIL import Image

            doc = fitz.open(path)
            n = len(doc) if max_pages is None else min(len(doc), max_pages)
            texts = []
            for i in range(n):
                pix = doc[i].get_pixmap(dpi=self.dpi)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(
                    tmp.name, "PNG"
                )
                r = self.process_image(tmp.name)
                os.unlink(tmp.name)
                texts.append(r.text if r.error is None else "")
            return OCRResult(
                text="\n\n".join(texts),
                engine=self.name,
                confidence=0.0,  # never invented
                processing_time=time.perf_counter() - start,
                pages=n,
                meta={
                    "lang": self.lang,
                    "psm": self.psm,
                    "dpi": self.dpi,
                    "tessdata_dir": self.tessdata_dir,
                },
            )
        except Exception as exc:
            return OCRResult(engine=self.name, error=f"{type(exc).__name__}: {exc}")
