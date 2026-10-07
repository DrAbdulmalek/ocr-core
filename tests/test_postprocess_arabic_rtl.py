"""Tests for the postprocess-layer Arabic RTL wrapper.

The wrapper must delegate verbatim to ``ocr_core.rtl_utils`` — these tests
pin that contract without duplicating rtl_utils' own test matrix.
"""
from __future__ import annotations

from ocr_core.postprocess.arabic_rtl import ArabicRTLPostProcessor
from ocr_core.rtl_utils import ArabicRTLFixer, RTLFixStats


def test_wrapper_delegates_to_rtl_utils_verbatim():
    text = "al salam o alaykom"
    fixer = ArabicRTLFixer()
    wrapper = ArabicRTLPostProcessor()
    assert wrapper.fix(text) == fixer.fix_text(text)


def test_wrapper_non_arabic_returned_normalized():
    wrapper = ArabicRTLPostProcessor()
    assert wrapper.fix("plain english line") == "plain english line"


def test_wrapper_should_fix_passthrough():
    wrapper = ArabicRTLPostProcessor()
    assert isinstance(wrapper.should_fix("hello"), bool)


def test_wrapper_threshold_property_matches_underlying():
    wrapper = ArabicRTLPostProcessor(reversal_threshold=0.5)
    assert wrapper.reversal_threshold == 0.5


def test_wrapper_fix_with_stats_returns_stats_payload():
    wrapper = ArabicRTLPostProcessor()
    fixed, stats = wrapper.fix_with_stats("hello world")
    assert isinstance(fixed, str)
    assert isinstance(stats, RTLFixStats)
    assert hasattr(stats, "reversal_ratio") and hasattr(stats, "changed")
