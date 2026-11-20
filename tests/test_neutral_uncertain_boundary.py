"""Regression tests for the neutral-vs-uncertain boundary.

Neutral means no correctness judgment was made by a helper; uncertain means
text was submitted for correctness classification but lacks sufficient
answer-scoped evidence. Neither state may be promoted to correct/incorrect.
"""
import pytest

from ocr_core.rules.engine import OCRProcessor


@pytest.fixture()
def proc():
    return OCRProcessor()


@pytest.mark.parametrize(
    "detected",
    [
        [],
        ["?"],
        ["⚠"],
        ["★"],
        ["†"],
    ],
)
def test_visual_helper_returns_neutral_when_no_classifying_marker_is_present(proc, detected):
    result = proc.apply_visual_rules("descriptive medical text", {"detected": detected})
    assert result["status"] == "neutral"
    assert result["label"] is None
    assert result["correct_markers"] == []
    assert result["incorrect_markers"] == []


def test_color_helper_without_color_regions_is_neutral(proc):
    result = proc.detect_color_with_legend("Descriptive text; no color regions supplied.", [])
    assert result["status"] == "neutral"
    assert result["regions"] == []
    assert result["label"] is None


@pytest.mark.parametrize(
    "text",
    [
        "A fluent answer with no visual evidence.",
        "Q1: first answer. Q2: second answer.",
        "Q1: answer.\n✓\nQ2: another answer.",
        "A medical X-ray finding with no correction evidence.",
        "Q1: answer. ✓ ✗",
    ],
)
def test_process_text_keeps_unresolved_correctness_uncertain(proc, text):
    result = proc.process_text(text)
    assert result["classification"] == "uncertain"
    assert result["correct"] == ""
    assert result["wrong"] == ""
    assert result["uncertain"].strip()


def test_ambiguous_x_is_uncertain_not_neutral_or_incorrect(proc):
    result = proc.apply_visual_rules("X-ray", {"detected": ["X"]})
    assert result["status"] == "uncertain"
    assert result["incorrect_markers"] == []
    assert result["ambiguous_markers"] == ["X"]


def test_explicit_positive_marker_can_classify_one_unambiguous_answer(proc):
    result = proc.process_text("Q1: verified answer. ✓")
    assert result["classification"] == "correct"
    assert "Q1" in result["correct"]
    assert result["wrong"] == ""
    assert result["uncertain"] == ""


@pytest.mark.parametrize("dose", ["1.5 mg", "0.25 mL", "2.75 mg/kg"])
def test_decimal_medical_values_do_not_create_false_answer_segments(proc, dose):
    # Decimal points inside clinical quantities are not sentence boundaries.
    result = proc.process_text(f"Q1: verified dosage {dose}. ✓")
    assert result["classification"] == "correct"
    assert "verified dosage" in result["correct"]
    assert result["uncertain"] == ""
