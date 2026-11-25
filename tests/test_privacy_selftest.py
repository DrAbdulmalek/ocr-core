"""Tests for privacy self-test verdict semantics."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "privacy_selftest.py"
SPEC = importlib.util.spec_from_file_location("privacy_selftest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_regex_only_never_reports_clean():
    verdict, code = MODULE._verdict(0, ner_requested=False, ner_loaded=False)
    assert verdict == "regex_only_incomplete_manual_review_required"
    assert code == 2


def test_requested_but_unavailable_ner_is_incomplete():
    verdict, code = MODULE._verdict(0, ner_requested=True, ner_loaded=False)
    assert verdict == "regex_only_incomplete_manual_review_required"
    assert code == 2


def test_active_local_ner_and_no_regex_residue_still_requires_manual_review():
    verdict, code = MODULE._verdict(0, ner_requested=True, ner_loaded=True)
    assert verdict == "no_detected_regex_residue_ner_active_manual_review_required"
    assert code == 0


def test_regex_residue_is_failure_even_if_ner_is_loaded():
    verdict, code = MODULE._verdict(2, ner_requested=True, ner_loaded=True)
    assert verdict == "regex_residue_detected"
    assert code == 1
