"""Unified document pipeline — six-stage clean-room design.

Clean-room provenance (see ``docs/research/abbyy_features.md``, decision D-004):
the *structure* below is extracted from publicly documented behaviour of
commercial document-processing products (stage names, layout-analysis modes,
selective re-recognition). No proprietary code, models, dictionaries or
weights were consulted or used. Only public documentation concepts were
modeled, then implemented from scratch with open engines.

Pipeline (public-documentation concept -> local implementation):

    PREPROCESS -> LAYOUT ANALYSIS -> RECOGNITION
        -> PAGE SYNTHESIS -> DOCUMENT SYNTHESIS -> EXPORT

Layout analysis modes (public-documentation concept -> local implementation):

    SIMPLE   text only — fastest
    COMPLEX  text + tables + pictures + headers/footers
    TABLES   table regions only
    AUTO     heuristically chooses between SIMPLE and COMPLEX

Policies preserved from ocr-core:

- Confidence ``0.0`` means "unknown" and is never invented (``engines.base``).
- Engines report failures through ``OCRResult.error``; this pipeline likewise
  reports stage failures through ``StageResult.error`` instead of raising.
- Raw recognized text is preserved verbatim; corrections/normalization are
  exposed in *separate* keys (``corrected_text``, ``normalized_text``),
  mirroring the ``ocr_service`` contract in consumers.

The core is dependency-light: without ``preprocess`` extras the PREPROCESS
stage is a documented passthrough, and layout regions default to a single
full-page text region. A ``layout_provider`` hook allows plugging the
deterministic ``ProjectionLayoutAnalyzer`` (or any future engine) without
changing this module.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ocr_core.engines.base import OCREngine, OCRResult
from ocr_core.telemetry import log_decision

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentPipeline",
    "LayoutMode",
    "PageResult",
    "PipelineResult",
    "PipelineStage",
    "RegionRecord",
    "StageResult",
]


class PipelineStage(str, Enum):
    """Six logical stages of the document pipeline."""

    PREPROCESS = "preprocess"
    LAYOUT = "layout"
    RECOGNITION = "recognition"
    PAGE_SYNTHESIS = "page_synthesis"
    DOCUMENT_SYNTHESIS = "document_synthesis"
    EXPORT = "export"


class LayoutMode(str, Enum):
    """Layout-analysis modes (public-documentation concepts)."""

    SIMPLE = "simple"
    COMPLEX = "complex"
    TABLES = "tables"
    AUTO = "auto"


#: Region kinds a layout provider may emit. ``text`` is the fallback kind.
REGION_KINDS = ("text", "table", "picture", "header", "footer")


@dataclass
class RegionRecord:
    """One layout region with its recognized text (if recognized)."""

    bbox: tuple[int, int, int, int]  # x, y, w, h in source-image pixel space
    kind: str = "text"
    text: str = ""
    engine: str = ""
    confidence: float = 0.0  # 0.0 = unknown — never a fabricated score
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "kind": self.kind,
            "text": self.text,
            "engine": self.engine,
            "confidence": self.confidence,
            "error": self.error,
            "meta": self.meta,
        }


@dataclass
class StageResult:
    """Outcome of a single pipeline stage for one page."""

    stage: PipelineStage
    ok: bool = True
    skipped: bool = False
    error: str | None = None
    duration: float = 0.0
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ok": self.ok,
            "skipped": self.skipped,
            "error": self.error,
            "duration": round(self.duration, 6),
            "detail": self.detail,
        }


@dataclass
class PageResult:
    """Per-page artifacts produced by the pipeline."""

    page_number: int
    text: str = ""
    corrected_text: str = ""
    normalized_text: str = ""
    corrections_applied: int = 0
    regions: list[RegionRecord] = field(default_factory=list)
    engine: str = ""
    confidence: float = 0.0  # 0.0 = unknown — never a fabricated score
    error: str | None = None
    stages: list[StageResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def stage(self, stage: PipelineStage) -> StageResult | None:
        for s in self.stages:
            if s.stage is stage:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "corrected_text": self.corrected_text,
            "normalized_text": self.normalized_text,
            "corrections_applied": self.corrections_applied,
            "regions": [r.to_dict() for r in self.regions],
            "engine": self.engine,
            "confidence": self.confidence,
            "error": self.error,
            "stages": [s.to_dict() for s in self.stages],
            "meta": self.meta,
        }


@dataclass
class PipelineResult:
    """Full document result: pages + stage trace + synthesis products."""

    source: str
    pages: list[PageResult] = field(default_factory=list)
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and all(p.error is None for p in self.pages)

    @property
    def text(self) -> str:
        """Document synthesis: page texts joined with blank lines."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def corrected_text(self) -> str:
        return "\n\n".join(p.corrected_text for p in self.pages if p.corrected_text)

    def stage_trace(self) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for p in self.pages:
            for s in p.stages:
                trace.append({"page": p.page_number, **s.to_dict()})
        return trace

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ok": self.ok,
            "error": self.error,
            "pages": [p.to_dict() for p in self.pages],
            "meta": self.meta,
        }


def _resolve_engine(engine: OCREngine | None) -> OCREngine | None:
    """Return an engine that is actually available, or None.

    Explicit engines are probed with ``available()``; without an explicit
    engine the router's low-profile order (Tesseract, EasyOCR, Paddle) is
    probed so the pipeline works on minimal installs.
    """
    if engine is not None:
        return engine if engine.available() else None
    from ocr_core.engines.easyocr import EasyOCREngine
    from ocr_core.engines.paddle import PaddleEngine
    from ocr_core.engines.tesseract import TesseractEngine

    for candidate in (TesseractEngine(), EasyOCREngine(), PaddleEngine()):
        try:
            if candidate.available():
                return candidate
        except Exception:  # probe must never crash the pipeline
            logger.debug("engine probe failed for %s", candidate.name, exc_info=True)
    return None


def _default_layout_provider(
    image_path: Path, mode: LayoutMode
) -> list[RegionRecord]:
    """Fallback layout: one full-page text region.

    Returns an empty bbox ``(0, 0, 0, 0)`` when pixel dimensions cannot be
    read without optional dependencies — the region still receives the
    whole-page recognition result.
    """
    width = height = 0
    try:  # pillow is part of the tesseract/benchmarks extras, not core
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        pass
    bbox = (0, 0, width, height) if width and height else (0, 0, 0, 0)
    kind = "text" if mode is not LayoutMode.TABLES else "table"
    return [RegionRecord(bbox=bbox, kind=kind, meta={"provider": "full_page_fallback"})]


class DocumentPipeline:
    """Six-stage document pipeline over pluggable OCR engines.

    Parameters
    ----------
    engine:
        Explicit engine instance. When omitted, the first available engine
        from the router's low profile (Tesseract -> EasyOCR -> Paddle) is used.
    layout_mode:
        ``SIMPLE`` / ``COMPLEX`` / ``TABLES`` / ``AUTO`` (default ``AUTO``).
    layout_provider:
        Callable ``(image_path, mode) -> list[RegionRecord]``. Defaults to a
        single full-page text region; pass ``ProjectionLayoutAnalyzer``-backed
        callables for region-aware runs.
    preprocess:
        Run the PREPROCESS stage via ``ocr_core.preprocess.pipeline.fix_scan``
        when the ``preprocess`` extra is installed; otherwise a documented
        passthrough is recorded.
    corrections:
        Optional ``ArabicMedicalCorrections``-like object exposing
        ``apply(text) -> (text, count)``.
    rtl_postprocess:
        Optional RTL/Arabic post-processor exposing ``fix(text) -> str``
        (e.g. ``postprocess.arabic_rtl.ArabicRTLPostProcessor``).
    """

    def __init__(
        self,
        engine: OCREngine | None = None,
        layout_mode: LayoutMode | str = LayoutMode.AUTO,
        layout_provider: Callable[[Path, LayoutMode], list[RegionRecord]] | None = None,
        preprocess: bool = True,
        corrections: Any | None = None,
        rtl_postprocess: Any | None = None,
    ) -> None:
        if isinstance(layout_mode, str):
            layout_mode = LayoutMode(layout_mode.lower())
        self.layout_mode = layout_mode
        self.layout_provider = layout_provider
        self.preprocess_enabled = preprocess
        self.corrections = corrections
        self.rtl_postprocess = rtl_postprocess
        self._engine = engine
        self._resolved: OCREngine | None = None

    # ------------------------------------------------------------------
    # stage implementations
    # ------------------------------------------------------------------
    def _stage_preprocess(self, image_path: Path) -> tuple[Path, StageResult]:
        """Preprocess stage: fix_scan when available, passthrough otherwise."""
        start = time.perf_counter()
        if not self.preprocess_enabled:
            return image_path, StageResult(
                stage=PipelineStage.PREPROCESS,
                skipped=True,
                duration=time.perf_counter() - start,
                detail={"reason": "preprocess disabled"},
            )
        try:
            from ocr_core.preprocess.pipeline import fix_scan
        except ImportError:
            return image_path, StageResult(
                stage=PipelineStage.PREPROCESS,
                skipped=True,
                duration=time.perf_counter() - start,
                detail={"reason": "preprocess extra not installed (passthrough)"},
            )
        try:
            report = fix_scan(image_path, output_path=None)
            out = report.get("output_path") or report.get("output") or image_path
            processed = Path(out) if out else image_path
            return processed, StageResult(
                stage=PipelineStage.PREPROCESS,
                duration=time.perf_counter() - start,
                detail={"report": {k: v for k, v in report.items() if k not in ("output", "output_path")}},
            )
        except Exception as exc:  # preprocessing must not kill recognition
            log_decision(
                decision="pipeline.preprocess_failed_passthrough",
                outcome="passthrough",
                reasons=[str(exc)],
                inputs={"image": str(image_path)},
            )
            return image_path, StageResult(
                stage=PipelineStage.PREPROCESS,
                ok=False,
                error=f"preprocess failed; passthrough used: {exc}",
                duration=time.perf_counter() - start,
            )

    def _effective_mode(self, regions: Sequence[RegionRecord]) -> LayoutMode:
        if self.layout_mode is not LayoutMode.AUTO:
            return self.layout_mode
        kinds = {r.kind for r in regions}
        if "table" in kinds and "text" in kinds:
            return LayoutMode.COMPLEX
        if "table" in kinds:
            return LayoutMode.TABLES
        return LayoutMode.SIMPLE

    def _stage_layout(self, image_path: Path) -> tuple[list[RegionRecord], StageResult]:
        start = time.perf_counter()
        provider = self.layout_provider or _default_layout_provider
        try:
            regions = list(provider(image_path, self.layout_mode))
        except Exception as exc:
            log_decision(
                decision="pipeline.layout_provider_failed",
                outcome="full_page_fallback",
                reasons=[str(exc)],
                inputs={"image": str(image_path)},
            )
            regions = _default_layout_provider(image_path, self.layout_mode)
            return regions, StageResult(
                stage=PipelineStage.LAYOUT,
                ok=False,
                error=f"layout provider failed; full-page fallback used: {exc}",
                duration=time.perf_counter() - start,
            )
        mode = self._effective_mode(regions)
        if mode is LayoutMode.TABLES:
            regions = [r for r in regions if r.kind == "table"] or regions
        return regions, StageResult(
            stage=PipelineStage.LAYOUT,
            duration=time.perf_counter() - start,
            detail={
                "mode_requested": self.layout_mode.value,
                "mode_effective": mode.value,
                "regions": len(regions),
            },
        )

    def _stage_recognition(
        self, image_path: Path, regions: list[RegionRecord]
    ) -> tuple[list[RegionRecord], OCRResult, StageResult]:
        start = time.perf_counter()
        engine = self._ensure_engine()
        if engine is None:
            error = (
                "no OCR engine available — install the [tesseract] extra "
                "or pass an explicit engine"
            )
            for r in regions:
                r.error = error
            empty = OCRResult(engine="", error=error)
            return regions, empty, StageResult(
                stage=PipelineStage.RECOGNITION,
                ok=False,
                error=error,
                duration=time.perf_counter() - start,
            )

        # Full-page shortcut: single text region from the fallback provider
        # is recognized in one engine call (cheaper than a crop round-trip).
        whole_page: OCRResult | None = None
        single = regions[0] if len(regions) == 1 else None
        if single is not None and single.kind == "text" and single.meta.get("provider") == "full_page_fallback":
            whole_page = engine.process_image(image_path)

        if whole_page is not None:
            regions[0].text = whole_page.text
            regions[0].engine = whole_page.engine
            regions[0].confidence = whole_page.confidence
            regions[0].error = whole_page.error
            return (
                regions,
                whole_page,
                StageResult(
                    stage=PipelineStage.RECOGNITION,
                    ok=whole_page.ok,
                    error=whole_page.error,
                    duration=time.perf_counter() - start,
                    detail={"engine": whole_page.engine, "regions": 1},
                ),
            )

        # region-wise recognition (ABBYY-style selective recognition concept)
        from ocr_core.regional import rerun_region

        agg_errors: list[str] = []
        for region in regions:
            if region.kind == "picture":
                continue  # pictures are not recognized
            result = rerun_region(image_path, region.bbox, engine=engine)
            region.text = result.text
            region.engine = result.engine
            region.confidence = result.confidence
            region.error = result.error
            if result.error:
                agg_errors.append(f"region {region.bbox}: {result.error}")
        combined_error = "; ".join(agg_errors) or None
        page_text = "\n".join(r.text for r in regions if r.text)
        combined = OCRResult(
            text=page_text,
            engine=engine.name,
            confidence=0.0,  # cross-region confidences are not comparable
            processing_time=time.perf_counter() - start,
            error=combined_error,
        )
        return regions, combined, StageResult(
            stage=PipelineStage.RECOGNITION,
            ok=combined_error is None,
            error=combined_error,
            duration=time.perf_counter() - start,
            detail={"engine": engine.name, "regions": len(regions)},
        )

    def _stage_page_synthesis(
        self,
        page: PageResult,
        regions: Sequence[RegionRecord],
        recognition: OCRResult,
    ) -> StageResult:
        """Rebuild the page text from regions in reading order (top-left)."""
        start = time.perf_counter()
        if recognition.error is not None and not recognition.text:
            page.text = ""
        elif len(regions) == 1:
            page.text = recognition.text
        else:
            ordered = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))
            page.text = "\n".join(r.text for r in ordered if r.text)
        if self.rtl_postprocess is not None and page.text:
            try:
                page.normalized_text = str(self.rtl_postprocess.fix(page.text))
            except Exception as exc:
                logger.debug("rtl postprocess failed: %s", exc)
        page.engine = recognition.engine
        page.confidence = recognition.confidence
        page.error = recognition.error
        return StageResult(
            stage=PipelineStage.PAGE_SYNTHESIS,
            ok=True,
            duration=time.perf_counter() - start,
            detail={"chars": len(page.text)},
        )

    def _stage_document_synthesis(self, page: PageResult) -> StageResult:
        """Apply optional corrections; raw text stays verbatim in ``text``."""
        start = time.perf_counter()
        if self.corrections is not None and page.text:
            try:
                corrected, count = self.corrections.apply(page.text)
                page.corrected_text = corrected
                page.corrections_applied = count
            except Exception as exc:
                return StageResult(
                    stage=PipelineStage.DOCUMENT_SYNTHESIS,
                    ok=False,
                    error=f"corrections failed: {exc}",
                    duration=time.perf_counter() - start,
                )
        return StageResult(
            stage=PipelineStage.DOCUMENT_SYNTHESIS,
            duration=time.perf_counter() - start,
            detail={"corrections_applied": page.corrections_applied},
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def _ensure_engine(self) -> OCREngine | None:
        if self._resolved is None:
            self._resolved = _resolve_engine(self._engine)
        return self._resolved

    def process_image(self, image_path: str | Path) -> PipelineResult:
        """Run the full pipeline on a single image page."""
        source = Path(image_path)
        result = PipelineResult(source=str(source))
        if not source.exists():
            result.error = f"source not found: {source}"
            return result

        page = PageResult(page_number=1)
        # PREPROCESS
        processed, pre = self._stage_preprocess(source)
        page.stages.append(pre)
        # LAYOUT
        regions, layout = self._stage_layout(processed)
        page.regions = regions
        page.stages.append(layout)
        # RECOGNITION
        regions, recognition, recog = self._stage_recognition(processed, regions)
        page.stages.append(recog)
        # PAGE SYNTHESIS
        page.stages.append(self._stage_page_synthesis(page, regions, recognition))
        # DOCUMENT SYNTHESIS
        page.stages.append(self._stage_document_synthesis(page))
        result.pages.append(page)
        result.meta["layout_mode"] = self._effective_mode(regions).value
        return result

    def process_pdf(self, pdf_path: str | Path, max_pages: int | None = None) -> PipelineResult:
        """Run the pipeline over a PDF via the engine's PDF path.

        Engines without PDF support return an ``OCRResult.error``; the
        pipeline surfaces it without raising.
        """
        source = Path(pdf_path)
        result = PipelineResult(source=str(source))
        if not source.exists():
            result.error = f"source not found: {source}"
            return result
        engine = self._ensure_engine()
        if engine is None:
            result.error = "no OCR engine available"
            return result
        base = engine.process_pdf(source, max_pages=max_pages)
        if base.error is not None and not base.text:
            result.error = base.error
            return result
        page_texts = [t for t in base.text.split("\f") if t.strip()] or [base.text]
        for idx, text in enumerate(page_texts, start=1):
            page = PageResult(page_number=idx, text=text, engine=base.engine)
            # optional post-processing parity with the image path
            if self.rtl_postprocess is not None and text:
                try:
                    page.normalized_text = str(self.rtl_postprocess.fix(text))
                except Exception as exc:
                    logger.debug("rtl postprocess failed: %s", exc)
            if self.corrections is not None and text:
                try:
                    corrected, count = self.corrections.apply(text)
                    page.corrected_text = corrected
                    page.corrections_applied = count
                except Exception as exc:
                    page.error = f"corrections failed: {exc}"
            page.stages.append(
                StageResult(stage=PipelineStage.PREPROCESS, skipped=True, detail={"reason": "pdf path — engine-owned"})
            )
            page.stages.append(
                StageResult(stage=PipelineStage.LAYOUT, skipped=True, detail={"reason": "pdf path — engine-owned"})
            )
            page.stages.append(
                StageResult(
                    stage=PipelineStage.RECOGNITION,
                    ok=base.error is None,
                    error=base.error,
                    detail={"engine": base.engine},
                )
            )
            page.stages.append(StageResult(stage=PipelineStage.PAGE_SYNTHESIS, detail={"chars": len(text)}))
            page.stages.append(
                StageResult(stage=PipelineStage.DOCUMENT_SYNTHESIS, detail={"corrections_applied": page.corrections_applied})
            )
            result.pages.append(page)
        return result

    # ------------------------------------------------------------------
    # EXPORT stage (document synthesis product -> serialization)
    # ------------------------------------------------------------------
    def export(self, result: PipelineResult, fmt: str = "json", out_dir: str | Path | None = None) -> str | Path:
        """Export a synthesized document.

        fmt:
            ``json`` — full structured result
            ``markdown`` — page texts with ``---`` separators
            ``txt`` — plain joined text
        out_dir:
            When given, writes ``<stem>.<ext>`` there and returns the path;
            otherwise returns the serialized string.
        """
        start = time.perf_counter()
        stem = Path(result.source).stem
        content: str
        ext = {"json": "json", "markdown": "md", "txt": "txt"}.get(fmt)
        if ext is None:
            raise ValueError(f"unsupported export format: {fmt!r} (json|markdown|txt)")
        if fmt == "json":
            content = json_dumps(result.to_dict())
        elif fmt == "markdown":
            parts = [f"# {stem}\n"]
            for page in result.pages:
                body = page.corrected_text or page.text
                parts.append(f"## Page {page.page_number}\n\n{body}\n")
            content = "\n".join(parts)
        else:
            content = result.corrected_text or result.text

        record_stage(result, PipelineStage.EXPORT, time.perf_counter() - start, {"format": fmt})
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            target = out_path / f"{stem}.{ext}"
            target.write_text(content, encoding="utf-8")
            return target
        return content


# ----------------------------------------------------------------------
# small stdlib helpers (kept module-level for testability)
# ----------------------------------------------------------------------
def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def record_stage(result: PipelineResult, stage: PipelineStage, duration: float, detail: dict) -> None:
    """Append an EXPORT-stage record to every page (document-level op)."""
    for page in result.pages:
        page.stages.append(
            StageResult(stage=stage, duration=duration, detail=dict(detail))
        )
