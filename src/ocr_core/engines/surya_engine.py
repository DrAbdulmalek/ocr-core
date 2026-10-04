"""Surya engine (optional extra: pip install .[surya]) — verified against surya-ocr 0.22.1.

Live-verified API surface (0.22.1 — differs from older docs):
- DetectionPredictor(images, batch_size=None, include_maps=False) -> List[TextDetectionResult]
- RecognitionPredictor(images, layout_results=None, *, full_page=None) -> List[PageOCRResult]
    NOTE: no ``langs`` argument anymore — the model is multilingual (Arabic +
    English included) and language selection is automatic in 0.22+.
- LayoutPredictor(images, target_image_sizes=None, max_tokens=None) -> List[LayoutResult]
- PageOCRResult.blocks: List[BlockOCRResult]  (NOT ``text_lines`` like pre-0.17 docs)
    BlockOCRResult: polygon / confidence / label / reading_order / html / skipped / error
- Block text lives in ``html``; we strip tags with a stdlib html.parser (no deps).

Architecture note (0.22.1): DetectionPredictor() is client-backed by default
(a shared server process); use ``DetectionPredictor.local()`` for a
process-local copy. We expose ``local_detectors`` to choose.

Policy (engines/base.py): failures are reported via ``OCRResult.error`` —
never raised. Confidence is left 0.0 unless the backend supplies real values —
never invented. PDF rasterization uses pypdfium2 (Apache/BSD) — not pymupdf
(AGPL) — per the license constraint in AGENTS.md.
"""
from __future__ import annotations

import logging
import os
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, List, Optional

from ocr_core.engines.base import OCRResult, OCREngine

logger = logging.getLogger(__name__)

_SURYA_TARGET = (0, 22)  # verified against 0.22.x


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML -> text (original, stdlib-only). Keeps block content readable."""

    _SKIP = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")
        elif tag == "td" or tag == "th":
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("p", "div", "tr", "li"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [ln.strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception as exc:  # defensive: never let a parsing quirk escape
        logger.debug("html strip failed (%s) — falling back to tag regex", exc)
        import re

        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()


class SuryaEngine(OCREngine):
    """Surya OCR engine — 90+ languages incl. Arabic, layout + reading order.

    Targets surya-ocr 0.22.x (live-verified). Older 0.x releases have a
    different API (langs argument, text_lines schema) and are NOT supported.
    """

    name = "surya"

    def __init__(
        self,
        langs_hint: Optional[List[str]] = None,
        use_layout: bool = False,
        local_detectors: bool = False,
    ):
        self.langs_hint = langs_hint  # advisory only in 0.22+ (kept for meta)
        self.use_layout = use_layout
        self.local_detectors = local_detectors

    # ------------------------------------------------------------------
    # availability + version guard
    # ------------------------------------------------------------------
    @staticmethod
    def is_available() -> bool:
        try:
            import surya  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _version_warning() -> Optional[str]:
        try:
            import surya

            ver = tuple(
                int(p) for p in getattr(surya, "__version__", "0").split(".")[:2]
            )
            if ver and ver < _SURYA_TARGET:
                return (
                    f"surya-ocr {getattr(surya, '__version__')} has an older API; "
                    f"this engine targets {(_SURYA_TARGET[0], _SURYA_TARGET[1])}.x"
                )
        except Exception:  # pragma: no cover — version probe is best-effort
            pass
        return None

    def _make_predictors(self) -> dict[str, Any]:
        from surya.recognition import RecognitionPredictor

        preds: dict[str, Any] = {"rec": RecognitionPredictor()}
        if self.local_detectors:
            from surya.detection import DetectionPredictor

            preds["det"] = DetectionPredictor.local()
        return preds

    # ------------------------------------------------------------------
    # image path
    # ------------------------------------------------------------------
    def process_image(self, image_path) -> OCRResult:
        start = time.perf_counter()
        path = str(image_path)
        if not os.path.isfile(path):
            return OCRResult(engine=self.name, error=f"file not found: {path}")
        if not self.is_available():
            return OCRResult(
                engine=self.name, error="surya-ocr not installed (pip install .[surya])"
            )
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            preds = self._make_predictors()
            pages = preds["rec"]([img])  # full-page mode when no layout given
            page = pages[0] if pages else None
            if page is None:
                return OCRResult(engine=self.name, error="surya returned no pages")

            blocks = sorted(
                (b for b in (page.blocks or []) if not getattr(b, "skipped", False)),
                key=lambda b: getattr(b, "reading_order", 0),
            )
            lines_meta = []
            texts: List[str] = []
            confs: List[float] = []
            for b in blocks:
                txt = _html_to_text(getattr(b, "html", "") or "")
                if not txt:
                    continue
                texts.append(txt)
                conf = getattr(b, "confidence", None)
                if conf is not None:
                    confs.append(float(conf))
                poly = getattr(b, "polygon", None) or []
                lines_meta.append(
                    {
                        "label": getattr(b, "label", ""),
                        "reading_order": getattr(b, "reading_order", 0),
                        "bbox": [c for pt in poly for c in pt][:4] if poly else None,
                        "confidence": conf,
                    }
                )

            # optional layout enrichment (separate predictor, adds region labels)
            layout_meta: List[dict] = []
            if self.use_layout:
                try:
                    from surya.layout import LayoutPredictor

                    lay = LayoutPredictor()([img])[0]
                    layout_meta = [
                        {
                            "label": getattr(lb, "label", ""),
                            "position": getattr(lb, "position", 0),
                            "bbox": list(getattr(lb, "polygon", [])[:1])[0]
                            + list(getattr(lb, "polygon", [])[2])
                            if getattr(lb, "polygon", None)
                            else None,
                        }
                        for lb in (lay.bboxes or [])
                    ]
                except Exception as exc:  # layout is optional — degrade honestly
                    logger.debug("layout enrichment failed: %s", exc)
                    layout_meta = []

            text = "\n".join(texts).strip()
            confidence = sum(confs) / len(confs) if confs else 0.0  # 0.0 = unknown
            meta: dict[str, Any] = {
                "langs_hint": self.langs_hint,
                "blocks": lines_meta,
                "n_blocks": len(texts),
                "layout": layout_meta,
            }
            vwarn = self._version_warning()
            if vwarn:
                meta["version_warning"] = vwarn
            return OCRResult(
                text=text,
                engine=self.name,
                confidence=confidence,
                processing_time=time.perf_counter() - start,
                meta=meta,
            )
        except Exception as exc:  # engine failure -> reported, never raised
            return OCRResult(engine=self.name, error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # pdf path — pypdfium2 only (license-clean), graceful if absent
    # ------------------------------------------------------------------
    def process_pdf(self, pdf_path, max_pages: Optional[int] = None) -> OCRResult:
        start = time.perf_counter()
        path = str(pdf_path)
        if not os.path.isfile(path):
            return OCRResult(engine=self.name, error=f"file not found: {path}")
        try:
            import pypdfium2 as pdfium
        except ImportError:
            return OCRResult(
                engine=self.name,
                error="pypdfium2 not installed (pip install pypdfium2) — pymupdf is intentionally not used (AGPL)",
            )
        try:
            from PIL import Image

            pdf = pdfium.PdfDocument(path)
            n = len(pdf) if max_pages is None else min(len(pdf), max_pages)
            texts = []
            for i in range(n):
                page = pdf[i]
                bitmap = page.render(scale=self._dpi_scale())
                img = bitmap.to_pil().convert("RGB")
                tmp = Path(path).with_suffix(f".surya_p{i}.png")
                img.save(tmp, "PNG")
                try:
                    r = self.process_image(str(tmp))
                finally:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                texts.append(r.text if r.error is None else "")
            return OCRResult(
                text="\n\n".join(texts),
                engine=self.name,
                confidence=0.0,  # per-page confidences vary — do not aggregate a fiction
                processing_time=time.perf_counter() - start,
                pages=n,
                meta={"langs_hint": self.langs_hint, "renderer": "pypdfium2"},
            )
        except Exception as exc:
            return OCRResult(engine=self.name, error=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _dpi_scale() -> float:
        # ~200 DPI render (72dpi base * 2.78 ≈ 200)
        return float(os.environ.get("SURYA_PDF_SCALE", "2.78"))
