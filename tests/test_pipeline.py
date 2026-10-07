"""Tests for the unified six-stage document pipeline (clean-room design).

Conventions honored:
- confidence 0.0 = unknown, never fabricated;
- failures surface via ``error`` fields, not exceptions;
- raw text stays verbatim; corrections land in separate keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ocr_core.engines.base import OCREngine, OCRResult
from ocr_core.pipeline import (
    DocumentPipeline,
    LayoutMode,
    PipelineResult,
    PipelineStage,
    RegionRecord,
)


class FakeEngine(OCREngine):
    """Deterministic engine: returns configured text, records calls."""

    name = "fake"

    def __init__(self, text: str = "مرحبا بالعالم hello", available: bool = True):
        self._text = text
        self._available = available
        self.calls: list[Path] = []

    def available(self) -> bool:
        return self._available

    def process_image(self, image_path) -> OCRResult:
        self.calls.append(Path(image_path))
        return OCRResult(text=self._text, engine=self.name, confidence=0.0)

    def process_pdf(self, pdf_path, max_pages=None) -> OCRResult:
        return OCRResult(text="page one\fpage two", engine=self.name, confidence=0.0)


class UnavailableEngine(FakeEngine):
    def __init__(self):
        super().__init__(available=False)


def _png(tmp_path: Path) -> Path:
    from PIL import Image

    p = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(p)
    return p


# ----------------------------------------------------------------------
# enums / dataclasses
# ----------------------------------------------------------------------
def test_pipeline_stage_enum_has_six_stages():
    assert len(PipelineStage) == 6
    assert [s.value for s in PipelineStage] == [
        "preprocess",
        "layout",
        "recognition",
        "page_synthesis",
        "document_synthesis",
        "export",
    ]


def test_layout_mode_values_match_public_concepts():
    assert {m.value for m in LayoutMode} == {"simple", "complex", "tables", "auto"}


def test_region_record_defaults_policy():
    r = RegionRecord(bbox=(1, 2, 3, 4))
    assert r.confidence == 0.0  # unknown — never invented
    assert r.kind == "text"
    assert r.to_dict()["bbox"] == [1, 2, 3, 4]


# ----------------------------------------------------------------------
# pipeline: happy path
# ----------------------------------------------------------------------
def test_process_image_full_flow_with_fake_engine(tmp_path):
    engine = FakeEngine("نص تجريبي sample")
    pipe = DocumentPipeline(engine=engine, preprocess=False)
    result = pipe.process_image(_png(tmp_path))

    assert result.ok and result.error is None
    assert result.pages[0].text == "نص تجريبي sample"
    assert result.pages[0].engine == "fake"
    assert result.pages[0].confidence == 0.0
    assert engine.calls, "engine must have been invoked once for full-page fallback"
    stages = {s.stage for s in result.pages[0].stages}
    assert {
        PipelineStage.PREPROCESS,
        PipelineStage.LAYOUT,
        PipelineStage.RECOGNITION,
        PipelineStage.PAGE_SYNTHESIS,
        PipelineStage.DOCUMENT_SYNTHESIS,
    } <= stages
    assert result.meta["layout_mode"] in {"simple", "complex", "tables", "auto"}


def test_layout_stage_records_effective_mode(tmp_path):
    pipe = DocumentPipeline(engine=FakeEngine(), preprocess=False)
    result = pipe.process_image(_png(tmp_path))
    layout = result.pages[0].stage(PipelineStage.LAYOUT)
    assert layout is not None
    assert layout.detail["mode_requested"] == "auto"
    assert layout.detail["mode_effective"] == "simple"


def test_missing_source_reports_error_not_raise(tmp_path):
    pipe = DocumentPipeline(engine=FakeEngine())
    result = pipe.process_image(tmp_path / "nope.png")
    assert result.error is not None and not result.pages


def test_unavailable_engine_reports_error(tmp_path):
    pipe = DocumentPipeline(engine=UnavailableEngine(), preprocess=False)
    result = pipe.process_image(_png(tmp_path))
    assert result.pages[0].error is not None
    recog = result.pages[0].stage(PipelineStage.RECOGNITION)
    assert recog is not None and recog.ok is False and recog.error


# ----------------------------------------------------------------------
# pipeline: layout modes + region-wise recognition
# ----------------------------------------------------------------------
def test_region_wise_recognition_uses_regional_module(tmp_path):
    engine = FakeEngine("منطقة")
    two_regions = [
        RegionRecord(bbox=(0, 0, 60, 20), kind="text"),
        RegionRecord(bbox=(0, 20, 60, 20), kind="text"),
    ]

    def provider(image_path, mode):
        return [r for r in two_regions]

    pipe = DocumentPipeline(engine=engine, preprocess=False, layout_provider=provider)
    result = pipe.process_image(_png(tmp_path))

    assert len(result.pages[0].regions) == 2
    assert result.pages[0].text == "منطقة\nمنطقة"
    recog = result.pages[0].stage(PipelineStage.RECOGNITION)
    assert recog.detail["regions"] == 2
    # multi-region confidences are not comparable — stays unknown
    assert result.pages[0].confidence == 0.0


def test_tables_mode_filters_to_table_regions(tmp_path):
    provider_regions = [
        RegionRecord(bbox=(0, 0, 60, 10), kind="text"),
        RegionRecord(bbox=(0, 10, 60, 20), kind="table"),
    ]
    pipe = DocumentPipeline(
        engine=FakeEngine(),
        preprocess=False,
        layout_mode=LayoutMode.TABLES,
        layout_provider=lambda p, m: list(provider_regions),
    )
    result = pipe.process_image(_png(tmp_path))
    kinds = {r.kind for r in result.pages[0].regions}
    assert kinds == {"table"}


def test_failing_layout_provider_falls_back_to_full_page(tmp_path):
    def broken_provider(image_path, mode):
        raise RuntimeError("layout engine exploded")

    pipe = DocumentPipeline(
        engine=FakeEngine("fallback text"),
        preprocess=False,
        layout_provider=broken_provider,
    )
    result = pipe.process_image(_png(tmp_path))
    layout = result.pages[0].stage(PipelineStage.LAYOUT)
    assert layout.ok is False and "fallback" in layout.error
    assert result.pages[0].text == "fallback text"


def test_preprocess_passthrough_recorded_when_extra_missing(tmp_path, monkeypatch):
    # Blocking the whole preprocess module deterministically simulates a
    # minimal install (works even when cv2/pillow are already imported by
    # other tests in the suite).
    monkeypatch.setitem(sys.modules, "ocr_core.preprocess.pipeline", None)
    pipe = DocumentPipeline(engine=FakeEngine(), preprocess=True)
    result = pipe.process_image(_png(tmp_path))
    pre = result.pages[0].stage(PipelineStage.PREPROCESS)
    assert pre.skipped is True
    assert "passthrough" in pre.detail["reason"] or "not installed" in pre.detail["reason"]


# ----------------------------------------------------------------------
# pipeline: post-processing hooks (verbatim raw policy)
# ----------------------------------------------------------------------
class _FakeCorrections:
    def apply(self, text: str):
        return text.replace("تجريبي", "بديل"), 1


class _FakeRTL:
    def fix(self, text: str) -> str:
        return "[rtl]" + text


def test_corrections_go_to_separate_key_raw_text_verbatim(tmp_path):
    pipe = DocumentPipeline(
        engine=FakeEngine("نص تجريبي"), preprocess=False, corrections=_FakeCorrections()
    )
    result = pipe.process_image(_png(tmp_path))
    page = result.pages[0]
    assert page.text == "نص تجريبي"  # raw preserved
    assert page.corrected_text == "نص بديل"
    assert page.corrections_applied == 1


def test_rtl_postprocess_hook_applies_to_normalized_text(tmp_path):
    pipe = DocumentPipeline(
        engine=FakeEngine("hello world"), preprocess=False, rtl_postprocess=_FakeRTL()
    )
    result = pipe.process_image(_png(tmp_path))
    page = result.pages[0]
    assert page.text == "hello world"
    assert page.normalized_text == "[rtl]hello world"


# ----------------------------------------------------------------------
# pipeline: PDF path
# ----------------------------------------------------------------------
def test_process_pdf_splits_form_feeds_into_pages(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    pipe = DocumentPipeline(engine=FakeEngine())
    result = pipe.process_pdf(pdf)
    assert result.ok
    assert [p.text for p in result.pages] == ["page one", "page two"]
    assert all(p.stage(PipelineStage.RECOGNITION) is not None for p in result.pages)


def test_process_pdf_missing_source(tmp_path):
    pipe = DocumentPipeline(engine=FakeEngine())
    result = pipe.process_pdf(tmp_path / "missing.pdf")
    assert result.error is not None


# ----------------------------------------------------------------------
# EXPORT stage
# ----------------------------------------------------------------------
def test_export_json_string(tmp_path):
    pipe = DocumentPipeline(engine=FakeEngine("بيانات"), preprocess=False)
    result = pipe.process_image(_png(tmp_path))
    payload = pipe.export(result, fmt="json")
    assert '"source"' in payload and "بيانات" in payload


def test_export_markdown_and_txt(tmp_path):
    pipe = DocumentPipeline(engine=FakeEngine("سطر"), preprocess=False)
    result = pipe.process_image(_png(tmp_path))
    md = pipe.export(result, fmt="markdown")
    assert md.startswith("# page") and "## Page 1" in md
    txt = pipe.export(result, fmt="txt")
    assert txt == "سطر"
    export_stage = result.pages[0].stage(PipelineStage.EXPORT)
    assert export_stage is not None


def test_export_to_out_dir_writes_file(tmp_path):
    pipe = DocumentPipeline(engine=FakeEngine("خروج"), preprocess=False)
    result = pipe.process_image(_png(tmp_path))
    target = pipe.export(result, fmt="txt", out_dir=tmp_path / "out")
    assert Path(target).exists() and Path(target).read_text(encoding="utf-8") == "خروج"


def test_export_unknown_format_raises():
    pipe = DocumentPipeline(engine=FakeEngine())
    with pytest.raises(ValueError):
        pipe.export(PipelineResult(source="x"), fmt="docx")


def test_stage_trace_and_to_dict_roundtrip_shapes(tmp_path):
    pipe = DocumentPipeline(engine=FakeEngine("hello"), preprocess=False)
    result = pipe.process_image(_png(tmp_path))
    trace = result.stage_trace()
    assert trace and trace[0]["page"] == 1
    dumped = result.to_dict()
    assert dumped["ok"] is True and dumped["pages"][0]["page_number"] == 1
