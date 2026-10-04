"""EasyOCR engine — ported from omni-medical-suite/src/ocr/easyocr_engine.py.

Optional extra: pip install .[easyocr]
Missing easyocr or a missing file is reported via OCRResult.error, never raised.
GPU defaults to False (reproducible CI / CPU hosts); pass gpu=True to match the
original omni constructor default.

Arabic RTL repair is applied per detected line (same as omni) via
``ocr_core.rtl_utils.ArabicRTLFixer`` — the NFKC-derived normalization map.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from ocr_core.engines.base import OCREngine, OCRResult
from ocr_core.rtl_utils import ArabicRTLFixer

logger = logging.getLogger(__name__)


class EasyOCREngine(OCREngine):
    """Wraps EasyOCR with Arabic/English defaults from omni, on the OCREngine protocol."""

    name = "easyocr"

    def __init__(
        self,
        languages: Optional[list[str]] = None,
        gpu: bool = False,
        batch_size: int = 4,
        enable_rtl_fix: bool = True,
    ) -> None:
        self.languages = languages if languages is not None else ["ar", "en"]
        self.gpu = gpu
        self.batch_size = batch_size
        self.enable_rtl_fix = enable_rtl_fix
        self._reader = None
        self._import_error: str | None = None
        self.rtl_fixer = ArabicRTLFixer()

    def available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def _engine(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError:
                self._import_error = "easyocr not installed (pip install .[easyocr])"
                raise
            self._reader = easyocr.Reader(
                lang_list=list(self.languages),
                gpu=self.gpu,
                verbose=False,
            )
        return self._reader

    # ------------------------------------------------------------------ OCREngine
    def process_image(self, image_path) -> OCRResult:
        path = Path(image_path)
        if not path.is_file():
            return OCRResult(engine=self.name, error=f"file not found: {path}")
        if not self.available():
            return OCRResult(engine=self.name, error="easyocr not installed (pip install .[easyocr])")
        try:
            from PIL import Image
        except ImportError:
            return OCRResult(engine=self.name, error="pillow not installed (pip install .[easyocr])")
        started = time.time()
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:  # unreadable/unsupported image
            return OCRResult(engine=self.name, error=f"cannot open image: {e}")
        extracted = self._extract(_pil_to_numpy(image))
        return self._to_result(extracted, started)

    def process_pdf(self, pdf_path, max_pages: Optional[int] = None) -> OCRResult:
        path = Path(pdf_path)
        if not path.is_file():
            return OCRResult(engine=self.name, error=f"file not found: {path}")
        try:
            import fitz  # pymupdf
        except ImportError:
            return OCRResult(engine=self.name, error="pymupdf not installed (pip install pymupdf)")
        started = time.time()
        try:
            doc = fitz.open(str(path))
        except Exception as e:
            return OCRResult(engine=self.name, error=f"cannot open pdf: {e}")
        page_count = min(doc.page_count, max_pages) if max_pages else doc.page_count
        texts, confs = [], []
        for i in range(page_count):
            page = doc[i]
            pix = page.get_pixmap(dpi=200)
            from PIL import Image
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            extracted = self._extract(_pil_to_numpy(image))
            if extracted.get("error"):
                return OCRResult(engine=self.name, error=extracted["error"],
                                 pages=page_count, processing_time=time.time() - started)
            texts.append(extracted.get("text", ""))
            if extracted.get("lines"):
                confs.append(_mean_confidence(extracted["lines"]))
        return OCRResult(
            text="\n".join(t for t in texts if t),
            engine=self.name,
            confidence=sum(confs) / len(confs) if confs else 0.0,
            processing_time=time.time() - started,
            pages=page_count,
        )

    # ------------------------------------------------------------------ internals
    def _extract(self, image) -> dict[str, Any]:
        """Port of omni EasyOCREngine.extract_text (RGB numpy array)."""
        try:
            reader = self._engine()
        except ImportError:
            return {"text": "", "lines": [], "num_lines": 0, "error": self._import_error}
        except Exception as e:  # model download / init failure must stay visible
            return {"text": "", "lines": [], "num_lines": 0, "error": f"easyocr init failed: {e}"}

        try:
            raw_results = reader.readtext(
                _ensure_rgb(image),
                paragraph=False,
                batch_size=self.batch_size,
            )
        except Exception as e:
            return {"text": "", "lines": [], "num_lines": 0, "error": f"easyocr readtext failed: {e}"}

        lines = []
        for bbox_points, text, confidence in raw_results:
            cleaned = text.strip()
            if self.enable_rtl_fix and cleaned:
                cleaned = self.rtl_fixer.fix_text(cleaned)
            lines.append({
                "text": cleaned,
                "bbox": [[int(p[0]), int(p[1])] for p in bbox_points],
                "confidence": round(float(confidence), 4),
            })
        lines.sort(key=lambda l: (l["bbox"][0][1], l["bbox"][0][0]))
        return {
            "text": "\n".join(l["text"] for l in lines if l["text"]),
            "lines": lines,
            "num_lines": len(lines),
        }

    def _to_result(self, extracted: dict[str, Any], started: float) -> OCRResult:
        lines = extracted.get("lines") or []
        return OCRResult(
            text=extracted.get("text", ""),
            engine=self.name,
            confidence=_mean_confidence(lines) if lines else 0.0,
            processing_time=time.time() - started,
            error=extracted.get("error"),
            meta={"num_lines": len(lines)},
        )


def _mean_confidence(lines: list[dict]) -> float:
    """Mean of real per-line confidences. Empty input → 0.0 (unknown, never invented)."""
    vals = [float(l.get("confidence", 0.0)) for l in lines]
    return sum(vals) / len(vals) if vals else 0.0


def _pil_to_numpy(image) -> "Any":
    import numpy as np
    return np.asarray(image)


def _ensure_rgb(image):
    """Port of omni _ensure_rgb: grayscale/RGBA/BGR-safe conversion to RGB."""
    import numpy as np
    if len(image.shape) == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    if image.shape[2] == 3:
        b_avg = float(np.mean(image[:, :, 0]))
        r_avg = float(np.mean(image[:, :, 2]))
        if b_avg > r_avg * 1.2:
            return image[:, :, ::-1]
    return image
