"""PaddleOCR engine — ported from omni-medical-suite/src/ocr/paddle_engine.py.

Optional extra: pip install .[paddle]
Missing paddleocr or a missing file is reported via OCRResult.error, never raised.
GPU defaults to False (reproducible CI / CPU hosts). Pass use_gpu=True to match
the original omni constructor default.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from ocr_core.engines.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)

# Arabic-optimized detection params — copied from omni PaddleOCREngine.DEFAULT_PARAMS
DEFAULT_PARAMS = {
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.5,
    "det_db_unclip_ratio": 1.8,
    "max_text_length": 800,
    "use_mp": True,
    "use_angle_cls": False,
    "lang": "ar",
    "show_log": False,
    "enable_mkldnn": True,
}

CUSTOM_MODEL_DIR = "models/paddle_handwriting"


class PaddleEngine(OCREngine):
    """Wraps PaddleOCR with the Arabic/medical detection settings from omni."""

    name = "paddle"

    def __init__(
        self,
        lang: str = "ar",
        use_gpu: bool = False,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.8,
        max_text_length: int = 800,
        use_mp: bool = True,
        dpi: int = 200,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.dpi = dpi
        self._ocr = None
        self._import_error: str | None = None
        self._params = {
            **DEFAULT_PARAMS,
            "det_db_thresh": det_db_thresh,
            "det_db_box_thresh": det_db_box_thresh,
            "det_db_unclip_ratio": det_db_unclip_ratio,
            "max_text_length": max_text_length,
            "use_mp": use_mp,
            "lang": lang,
            "use_gpu": use_gpu,
        }

    def available(self) -> bool:
        if self._ocr is not None:
            return True
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return False
        return True

    def _engine(self):
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            self._import_error = "paddleocr not installed (pip install .[paddle])"
            raise exc
        init_params = {**self._params, **self._custom_model_config()}
        self._ocr = PaddleOCR(**init_params)
        return self._ocr

    def process_image(self, image_path) -> OCRResult:
        start = time.perf_counter()
        path = str(image_path)
        if not os.path.isfile(path):
            return OCRResult(engine=self.name, error=f"file not found: {path}")
        if not self.available():
            return OCRResult(
                engine=self.name,
                error="paddleocr not installed (pip install .[paddle])",
            )
        try:
            from PIL import Image
            import numpy as np

            image = np.array(Image.open(path).convert("RGB"))
            extracted = self._extract(image)
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                error=f"{type(exc).__name__}: {exc}",
                processing_time=time.perf_counter() - start,
            )
        if extracted.get("error"):
            return OCRResult(
                engine=self.name,
                error=str(extracted["error"]),
                processing_time=time.perf_counter() - start,
            )
        conf = _mean_confidence(extracted.get("lines") or [])
        return OCRResult(
            text=(extracted.get("text") or "").strip(),
            engine=self.name,
            confidence=conf,
            processing_time=time.perf_counter() - start,
            meta={
                "lang": self.lang,
                "num_lines": extracted.get("num_lines", 0),
                "lines": extracted.get("lines") or [],
                "source": "omni-medical-suite/src/ocr/paddle_engine.py",
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
            return OCRResult(
                engine=self.name,
                error="pymupdf not installed (pip install pymupdf)",
            )
        try:
            import numpy as np

            doc = fitz.open(path)
            n = len(doc) if max_pages is None else min(len(doc), max_pages)
            texts: list[str] = []
            confs: list[float] = []
            for i in range(n):
                pix = doc[i].get_pixmap(dpi=self.dpi)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    arr = arr[:, :, :3]
                elif pix.n == 1:
                    arr = np.stack([arr[:, :, 0]] * 3, axis=-1)
                extracted = self._extract(arr)
                texts.append(extracted.get("text") or "")
                confs.append(_mean_confidence(extracted.get("lines") or []))
            return OCRResult(
                text="\n\n".join(texts).strip(),
                engine=self.name,
                confidence=sum(confs) / len(confs) if confs else 0.0,
                processing_time=time.perf_counter() - start,
                pages=n,
                meta={"lang": self.lang, "dpi": self.dpi},
            )
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                error=f"{type(exc).__name__}: {exc}",
                processing_time=time.perf_counter() - start,
            )

    def _extract(self, image) -> dict[str, Any]:
        """Port of omni PaddleOCREngine.extract_text (RGB numpy array)."""
        try:
            engine = self._engine()
        except ImportError:
            return {"text": "", "lines": [], "num_lines": 0, "error": self._import_error}
        rgb = _ensure_rgb(image)
        raw_results = engine.ocr(rgb, cls=False)
        if not raw_results or raw_results[0] is None:
            return {"text": "", "lines": [], "num_lines": 0, "engine": self.name}
        lines: list[dict] = []
        for item in raw_results[0]:
            bbox_points = item[0]
            text_info = item[1]
            text = text_info[0].strip() if text_info[0] else ""
            confidence = float(text_info[1]) if len(text_info) > 1 else 0.0
            bbox_list = [[round(p[0]), round(p[1])] for p in bbox_points]
            lines.append(
                {"text": text, "bbox": bbox_list, "confidence": round(confidence, 4)}
            )
        lines.sort(key=lambda row: (row["bbox"][0][1], row["bbox"][0][0]))
        full_text = "\n".join(line["text"] for line in lines if line["text"])
        return {
            "text": full_text,
            "lines": lines,
            "num_lines": len(lines),
            "engine": self.name,
        }

    def _custom_model_config(self) -> dict[str, str]:
        custom_dir = _find_custom_model_dir()
        if custom_dir is None:
            return {}
        config: dict[str, str] = {}
        for kind, key in (("det", "det_model_dir"), ("rec", "rec_model_dir"), ("cls", "cls_model_dir")):
            d = os.path.join(custom_dir, kind)
            if os.path.isdir(d):
                config[key] = d
                logger.info("custom paddle model: %s", d)
        return config


def _mean_confidence(lines: list[dict]) -> float:
    vals = [float(x["confidence"]) for x in lines if x.get("text") and "confidence" in x]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def _find_custom_model_dir() -> str | None:
    env = os.environ.get("OCR_CORE_PADDLE_MODEL_DIR")
    search = []
    if env:
        search.append(Path(env))
    search.extend(
        [
            Path(CUSTOM_MODEL_DIR).resolve(),
            Path.cwd() / CUSTOM_MODEL_DIR,
        ]
    )
    for path in search:
        if path.is_dir():
            return str(path)
    return None


def _ensure_rgb(image):
    """Port of omni PaddleOCREngine._ensure_rgb."""
    import numpy as np

    if len(image.shape) == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    if image.shape[2] == 3:
        b_avg = float(np.mean(image[:, :, 0]))
        r_avg = float(np.mean(image[:, :, 2]))
        if b_avg > r_avg * 1.15:
            return image[:, :, ::-1]
    return image
