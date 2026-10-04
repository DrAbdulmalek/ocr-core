"""Tests for ocr_core.engines.ensemble (ported from omni, protocol-adapted).

Engines are fakes implementing the OCREngine protocol — no heavy deps.
"""
import pytest

from ocr_core.engines.base import OCREngine, OCRResult
from ocr_core.engines.ensemble import EnsembleEngine, _validity_ratio


class FakeEngine(OCREngine):
    def __init__(
        self,
        name,
        text="",
        confidence=0.0,
        error=None,
        available=True,
        raises=False,
    ):
        self.name = name
        self._text = text
        self._confidence = confidence
        self._error = error
        self._available = available
        self._raises = raises

    def available(self):
        return self._available

    def process_image(self, image_path):
        if self._raises:
            raise RuntimeError("boom")
        return OCRResult(
            text=self._text,
            engine=self.name,
            confidence=self._confidence,
            error=self._error,
        )


def test_validity_ratio_counts_arabic_and_spaces():
    arabic = "مرحبا بالعالم"
    assert _validity_ratio(arabic) == 1.0
    mixed = "abc اسد ###"
    # abc(3) + space + اسد(3) + space = 8 valid of 11
    assert _validity_ratio(mixed) == pytest.approx(8 / 11)
    assert _validity_ratio("") == 0.0


def test_no_engines_registered_is_fail_visible():
    ens = EnsembleEngine()
    assert not ens.available()
    r = ens.process_image("x.png")
    assert r.error == "no engines registered"
    assert r.confidence == 0.0


def test_available_aggregates_any_engine():
    ens = EnsembleEngine({"a": FakeEngine("a", available=False)})
    assert not ens.available()
    ens.register("b", FakeEngine("b", available=True))
    assert ens.available()


def test_unavailable_engine_captured_as_error():
    ens = EnsembleEngine(
        {
            "dead": FakeEngine("dead", available=False),
            "live": FakeEngine("live", text="hello", confidence=0.9),
        }
    )
    r = ens.process_image("x.png")
    assert r.error is None
    assert r.text == "hello"
    assert r.meta["per_engine"]["dead"]["error"] is not None
    assert r.meta["per_engine"]["live"]["error"] is None


def test_raising_engine_does_not_abort_ensemble():
    ens = EnsembleEngine(
        {
            "boom": FakeEngine("boom", raises=True),
            "good": FakeEngine("good", text="fine", confidence=0.8),
        }
    )
    r = ens.process_image("x.png")
    assert r.error is None
    assert r.text == "fine"
    assert r.meta["per_engine"]["boom"]["error"] == "boom"


def test_composite_prefers_higher_confidence_times_length():
    # 0.9 * 50 = 45  vs  0.5 * 200 = 100 -> the longer, less confident wins
    ens = EnsembleEngine(
        {
            "short": FakeEngine("short", text="a" * 50, confidence=0.9),
            "long": FakeEngine("long", text="b" * 200, confidence=0.5),
        }
    )
    r = ens.process_image("x.png")
    assert r.engine == "ensemble:long"
    assert r.confidence == 0.5


def test_unknown_confidence_tier_uses_length_validity_not_zero():
    # All confidences are 0.0 (unknown) — winner must still be chosen and
    # the reported confidence must remain 0.0 (never fabricated).
    ens = EnsembleEngine(
        {
            "small": FakeEngine("small", text="abc"),
            "large": FakeEngine("large", text="w" * 30),
        }
    )
    r = ens.process_image("x.png")
    assert r.engine == "ensemble:large"
    assert r.confidence == 0.0
    assert r.meta["strategy"] == "length-validity (confidence unknown)"


def test_confident_engines_outrank_unknown_ones():
    ens = EnsembleEngine(
        {
            "known": FakeEngine("known", text="hi", confidence=0.2),
            "unknown": FakeEngine("unknown", text="x" * 500),
        }
    )
    r = ens.process_image("x.png")
    assert r.engine == "ensemble:known"


def test_error_penalty_kicks_in_only_when_text_exists():
    # error engine with text: penalized x0.1 -> clean engine wins
    ens = EnsembleEngine(
        {
            "err": FakeEngine("err", text="z" * 100, confidence=0.9, error="partial"),
            "clean": FakeEngine("clean", text="y" * 100, confidence=0.85),
        }
    )
    r = ens.process_image("x.png")
    assert r.engine == "ensemble:clean"
    assert r.meta["per_engine"]["err"]["error"] == "partial"


def test_all_engines_failed_is_fail_visible():
    ens = EnsembleEngine(
        {
            "a": FakeEngine("a", error="dead"),
            "b": FakeEngine("b", raises=True),
        }
    )
    r = ens.process_image("x.png")
    assert r.error == "all engines failed"
    assert r.confidence == 0.0


def test_meta_contains_per_engine_breakdown():
    ens = EnsembleEngine(
        {
            "a": FakeEngine("a", text="hello", confidence=0.7),
            "b": FakeEngine("b", text="", error="nope"),
        }
    )
    r = ens.process_image("x.png")
    assert set(r.meta["per_engine"]) == {"a", "b"}
    assert r.meta["per_engine"]["a"]["chars"] == 5
    assert r.meta["per_engine"]["a"]["confidence"] == 0.7
    assert r.meta["winner"] == "a"


def test_comparison_table_renders_all_engines():
    ens = EnsembleEngine(
        {
            "a": FakeEngine("a", text="hello", confidence=0.7),
            "b": FakeEngine("b", error="dead"),
        }
    )
    table = ens.comparison_table("x.png")
    assert "a" in table and "b" in table
    assert "ok" in table and "ERR" in table
    # a has real confidence 0.7; b errored with no text -> conf shown as N/A
    assert table.count("N/A") == 1


def test_process_pdf_is_explicitly_unsupported():
    ens = EnsembleEngine({"a": FakeEngine("a", text="x")})
    r = ens.process_pdf("doc.pdf")
    assert r.error is not None
    assert "PDF" in r.error
