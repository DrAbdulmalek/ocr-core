# tests/test_evidence_scope_adversarial.py
"""Adversarial regressions for the visual-evidence scope contract (ocr-core).

Invariants exercised (see ocr_core/charter_v2.yaml):
  1  text_verbatim is preserved verbatim through normalization/classification
  2  text_normalized stays a separate representation
  3  text cleanliness / character ratios are diagnostic only, never proof
  4  ambiguous markers (X/x) are never treated as certain answers
  5  a training label is promoted only with confidence + evidence + source + explicit confirmation
  6  an uncertain case stays uncertain unless a valid evidence rule resolves it
  7  a marker attached to one answer does not classify unrelated answers or the whole document
  8  evidence from one processing call does not leak into a later call or document
  9  evidence is scoped to the text or region it actually supports
 10  existing public API / serialized keys stay backward compatible
"""
import pytest

from ocr_core.rules.engine import OCRProcessor

PUBLIC_KEYS = {
    "markdown", "text_verbatim", "text_normalized", "correct", "wrong",
    "uncertain", "classification", "ratio", "metadata",
}


@pytest.fixture()
def proc():
    return OCRProcessor()


# ---------- single answer, clear positive marker ----------
def test_clear_positive_marker_single_answer(proc):
    result = proc.process_text("Q1: نص الاجابة الصحيحة. \u2713")
    assert result["classification"] == "correct"
    assert "نص الاجابة" in result["correct"]
    assert result["wrong"] == ""


# ---------- single answer, clear negative marker ----------
def test_clear_negative_marker_single_answer(proc):
    result = proc.process_text("Q1: ترجمة خاطئة. \u2717")
    assert result["classification"] == "incorrect"
    assert "ترجمة خاطئة" in result["wrong"]
    assert result["correct"] == ""


# ---------- one marked answer among several ----------
def test_one_marked_answer_does_not_certify_its_neighbours(proc):
    page = (
        "Q1: نص الإجابة الاولى.\n"
        "Q2: نص الإجابة الثانيه. \u2713.\n"
        "Q3: نص الإجابة الثالثة."
    )
    result = proc.process_text(page)
    assert "Q2" in result["correct"]
    assert "Q1" not in result["correct"]
    assert "Q3" not in result["correct"]
    assert "Q1" in result["uncertain"] and "Q3" in result["uncertain"]
    assert result["classification"] == "uncertain"


# ---------- positive and negative markers at different locations ----------
def test_mixed_polarity_markers_route_to_separate_buckets(proc):
    page = (
        "Q1: نص الإجابة الاولى. \u2713.\n"
        "Q2: نص الإجابة الثانيه.\n"
        "Q3: نص الإجابة الثالثة. \u2717."
    )
    result = proc.process_text(page)
    assert "Q1" in result["correct"] and "Q3" in result["wrong"]
    assert "Q2" in result["uncertain"]
    assert result["classification"] == "uncertain"


# ---------- marked document followed by unmarked document, same engine ----------
def test_marked_then_unmarked_document_same_engine(proc):
    marked = proc.process_text("Q1: نص مؤكد صحيح. \u2713")
    assert marked["classification"] == "correct"
    unmarked = proc.process_text("Q1: نص عادي بلا اي دليل بصري.")
    assert unmarked["classification"] == "uncertain"
    assert unmarked["correct"] == "" and unmarked["wrong"] == ""


# ---------- several sequential calls on one instance ----------
def test_sequential_calls_do_not_accumulate_evidence(proc):
    for _ in range(3):
        proc.process_text("Q1: نص مؤكد صحيح. \u2713")
    for _ in range(3):
        result = proc.process_text("Q1: نص بلا دليل.")
        assert result["classification"] == "uncertain"
        assert result["correct"] == ""


# ---------- ambiguous X / x ----------
def test_ambiguous_uppercase_x_is_not_a_negative_answer(proc):
    result = proc.process_text("السطر الاول يذكر X-ray صراحة.\nالسطر الثاني عادي.")
    assert result["classification"] == "uncertain"
    assert result["wrong"] == ""
    assert "X" not in result["metadata"]["markers"]


def test_ambiguous_lowercase_x_is_not_a_negative_answer(proc):
    result = proc.process_text("متغير x في المعادلة.\nسطر ثان عادي.")
    assert result["classification"] == "uncertain"
    assert result["wrong"] == ""


# ---------- a marker that cannot be associated with an answer ----------
def test_marker_without_associable_answer_stays_uncertain(proc):
    result = proc.process_text("Q1: نص الاجابة الاولى.\n\u2713\nQ2: نص الاجابة الثانية.")
    assert result["classification"] == "uncertain"
    assert "Q1" not in result["correct"] and "Q2" not in result["correct"]


# ---------- missing / ambiguous legend ----------
def test_color_without_legend_keeps_semantics_uncertain(proc):
    rec = proc.build_training_record(
        source="doc", correct="نص سليم", incorrect="",
        evidence={"color": "red", "label": "negative_example",
                  "confidence": "high", "evidence_confirmed": True,
                  "source_reference": "doc.pdf#p1", "legend_found": False},
    )
    assert rec["status"] == "uncertain"


# ---------- missing source reference ----------
def test_missing_source_reference_blocks_promotion(proc):
    rec = proc.build_training_record(
        source="doc", correct="نص سليم", incorrect="",
        evidence={"visual_marker": "\u2713", "label": "positive_example",
                  "confidence": "high", "evidence_confirmed": True},
    )
    assert rec["status"] == "uncertain"
    assert rec["training_eligible"] is False
    assert rec["positive_example"] is None


# ---------- missing explicit confirmation ----------
def test_missing_explicit_confirmation_blocks_promotion(proc):
    rec = proc.build_training_record(
        source="doc", correct="نص سليم", incorrect="",
        evidence={"visual_marker": "\u2713", "label": "positive_example",
                  "confidence": "high", "source_reference": "doc.pdf#p1"},
    )
    assert rec["status"] == "uncertain"
    assert rec["training_eligible"] is False


# ---------- conflicting evidence ----------
def test_conflicting_evidence_in_one_scope_is_uncertain(proc):
    result = proc.process_text("Q1: نص الاجابة. \u2713 \u2717")
    assert result["classification"] == "uncertain"
    assert "Q1" in result["uncertain"]


# ---------- verbatim preservation through normalization ----------
def test_verbatim_is_preserved_through_normalization_and_classification(proc):
    raw = "التَّشخيص ١٢٣ ملم. \u2713"
    result = proc.process_text(raw)
    assert result["text_verbatim"] == raw
    assert result["text_normalized"] == result["markdown"]
    assert result["text_verbatim"] != result["text_normalized"]


# ---------- no line boundaries available ----------
def test_single_line_multi_answer_with_marker_is_not_whole_document_correct(proc):
    """OCR output without line breaks: one marker must not certify every answer."""
    blob = "Q1: الجواب الاول. Q2: الجواب الثاني. \u2713"
    result = proc.process_text(blob)
    assert result["classification"] == "uncertain", (
        "علامة واحدة في سطر واحد بلا حدود أسطر لا تصادق على كل الإجابات"
    )
    assert "الجواب الاول" not in result["correct"] or "الجواب الثاني" in result["uncertain"]


def test_single_line_without_answer_markers_is_not_whole_document_correct(proc):
    blob = "الجواب الاول. الجواب الثاني. \u2713"
    result = proc.process_text(blob)
    assert result["classification"] == "uncertain"


# ---------- public API stability ----------
def test_public_result_schema_stable(proc):
    """Invariant 10: the serialized result keeps the documented keys."""
    result = proc.process_text("نص. \u2713")
    assert PUBLIC_KEYS.issubset(result.keys())
    assert {"rules_file", "rules_applied", "markers", "classification"} <= set(result["metadata"].keys())
