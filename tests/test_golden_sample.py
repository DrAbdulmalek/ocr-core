"""Tests for the unified golden sample (docs/09 §6 item 1).

Unit tests are engine-free (stub engines) so CI — which installs only
``.[preprocess,benchmarks,golden]`` — stays deterministic and fast.
Real-engine integration tests self-skip when the tesseract binary or the
traineddata languages are missing.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_core.benchmarks.golden import (
    CONDITIONS,
    MIN_PAIRS,
    aggregate,
    apply_variant,
    golden_pairs,
    render_pair,
    to_markdown,
)
from ocr_core.benchmarks.golden.runner import run_golden_benchmark
from ocr_core.engines.base import OCRResult
from ocr_core.postprocess.normalization import arabic_strong_normalize

# ─── dataset integrity ────────────────────────────────────────────────────────


def test_dataset_meets_minimum_and_languages():
    pairs = golden_pairs()
    assert len(pairs) >= MIN_PAIRS
    langs = {p.lang for p in pairs}
    assert {"ar", "en"} <= langs
    # every pair has substantive ground truth
    for p in pairs:
        assert len(p.text) >= 40, f"{p.id} text too short"
        assert p.id and p.seed > 0 and p.font_px >= 30


def test_dataset_ids_unique_and_conditions_covered():
    pairs = golden_pairs()
    ids = [p.id for p in pairs]
    assert len(ids) == len(set(ids)), "pair ids must be unique"
    conds = {p.condition for p in pairs}
    assert conds == set(CONDITIONS), f"all conditions required, got {conds}"
    # both languages must exercise the shadow case (the Otsu-collapse proof)
    for lang in ("ar", "en"):
        assert any(p.lang == lang and p.condition == "shadow" for p in pairs)


# ─── renderer determinism + validity ─────────────────────────────────────────


def _page_bytes(pair):
    img = render_pair(pair)
    return np.asarray(img).tobytes(), img


def test_render_is_deterministic():
    pair = golden_pairs()[0]
    b1, img1 = _page_bytes(pair)
    b2, img2 = _page_bytes(pair)
    assert b1 == b2
    assert img1.size == img2.size


def test_render_produces_readable_page():
    pair = next(p for p in golden_pairs() if p.condition == "clean")
    img = render_pair(pair)
    arr = np.asarray(img)
    assert arr.ndim == 2 and arr.dtype == np.uint8
    # page must contain both paper and ink
    assert arr.max() >= 230, "no paper background"
    assert (arr < 80).sum() > 500, "no ink pixels rendered"
    assert img.width >= 1000, "page too narrow for A4-like OCR"


def test_render_rejects_unknown_condition():
    pair = golden_pairs()[0]
    bad = type(pair)(**{**pair.__dict__, "condition": "lava"})
    with pytest.raises(ValueError):
        render_pair(bad)


# ─── variant application ─────────────────────────────────────────────────────


def test_apply_variant_raw_is_identity():
    pair = next(p for p in golden_pairs() if p.condition == "clean")
    arr = np.asarray(render_pair(pair))
    out = apply_variant(arr, "raw")
    assert out is arr


def test_apply_variants_return_binary_images():
    pair = next(p for p in golden_pairs() if p.condition == "shadow")
    arr = np.asarray(render_pair(pair))
    for v in ("otsu", "sauvola", "auto"):
        out = apply_variant(arr, v)
        assert out.dtype == np.uint8
        assert set(np.unique(out)) <= {0, 255}, f"{v} must be binary"
        assert (out == 0).sum() > 0 and (out == 255).sum() > 0


def test_apply_variant_rejects_unknown():
    with pytest.raises(ValueError):
        apply_variant(np.zeros((10, 10), dtype=np.uint8), "magic")


# ─── runner with stub engines (no real OCR dependency) ───────────────────────


class _StubEngine:
    def __init__(self, name: str, respond_with):
        self.name = name
        self._respond = respond_with
        self.seen_paths: list[str] = []

    def process_image(self, image_path) -> OCRResult:
        self.seen_paths.append(str(image_path))
        return self._respond(Path(image_path))


def test_runner_stub_perfect_and_garbled(tmp_path):
    pair = golden_pairs()[0]
    norm_ref = arabic_strong_normalize(pair.text)

    perfect = _StubEngine(
        "stub-perfect", lambda p: OCRResult(text=pair.text, engine="stub-perfect")
    )
    garbled = _StubEngine(
        "stub-garbled",
        lambda p: OCRResult(text="نص مختلف تماما عن الحقيقة الأرضية", engine="stub-garbled"),
    )

    result = run_golden_benchmark(
        engines=[perfect, garbled], pairs=[pair], variants=("raw",), workdir=tmp_path
    )
    assert result["meta"]["n_pairs"] == 1
    rows = result["rows"]
    assert len(rows) == 2

    r_perfect = next(r for r in rows if r["engine"] == "stub-perfect")
    assert r_perfect["ok"] and r_perfect["cer"] == 0.0 and r_perfect["wer"] == 0.0
    # normalization is applied to BOTH sides — ground truth is normalized too
    assert r_perfect["ref_len"] == len(norm_ref)
    assert r_perfect["hyp_len"] == len(norm_ref)

    r_garbled = next(r for r in rows if r["engine"] == "stub-garbled")
    assert r_garbled["ok"] and r_garbled["cer"] > 0.5


def test_runner_stub_engine_error_is_a_row_not_a_crash(tmp_path):
    pair = golden_pairs()[0]

    def _boom(path):
        return OCRResult(engine="stub-broken", error="EngineCrash: simulated")

    broken = _StubEngine("stub-broken", _boom)
    result = run_golden_benchmark(
        engines=[broken], pairs=[pair], variants=("raw", "otsu"), workdir=tmp_path
    )
    assert len(result["rows"]) == 2
    for r in result["rows"]:
        assert not r["ok"] and "EngineCrash" in r["error"] and r["cer"] is None


def test_runner_normalization_makes_equivalences_equal(tmp_path):
    """Same text with different alef/teh-marbuta forms → CER 0 after normalize."""
    pair = next(p for p in golden_pairs() if p.lang == "en")
    sloppy = pair.text.replace("a", "á").replace("e", "é")  # harmless ASCII twist
    engine = _StubEngine("stub-eq", lambda p: OCRResult(text=pair.text, engine="stub-eq"))
    result = run_golden_benchmark(
        engines=[engine], pairs=[pair], variants=("raw",), workdir=tmp_path
    )
    row = result["rows"][0]
    assert row["ok"] and row["cer"] == 0.0
    assert sloppy != pair.text  # sanity: the twist was real


def test_runner_writes_variant_images_and_validates_variants(tmp_path):
    pair = golden_pairs()[0]
    engine = _StubEngine("stub", lambda p: OCRResult(text=pair.text, engine="stub"))
    result = run_golden_benchmark(
        engines=[engine], pairs=[pair], variants=("raw", "auto"), workdir=tmp_path
    )
    for v in ("raw", "auto"):
        assert (tmp_path / f"{pair.id}.{v}.png").is_file()
    assert (tmp_path / f"{pair.id}.png").is_file()
    assert result["meta"]["variants"] == ["raw", "auto"]

    with pytest.raises(ValueError):
        run_golden_benchmark(engines=[engine], pairs=[pair], variants=("nope",), workdir=tmp_path)


def test_runner_requires_engines(tmp_path):
    with pytest.raises(ValueError):
        run_golden_benchmark(engines=[], pairs=[golden_pairs()[0]], workdir=tmp_path)


# ─── aggregation + report ────────────────────────────────────────────────────


def _synthetic_rows():
    rows = []
    for engine in ("alpha", "beta"):
        for variant in ("raw", "auto"):
            for lang in ("ar", "en"):
                cer = 0.10 if engine == "alpha" else 0.30
                if variant == "auto":
                    cer = cer * 0.5
                if lang == "en":
                    cer = cer * 0.8
                rows.append(
                    {
                        "pair_id": "x-001",
                        "lang": lang,
                        "condition": "clean",
                        "engine": engine,
                        "variant": variant,
                        "ok": True,
                        "error": None,
                        "cer": cer,
                        "wer": cer * 2,
                        "substitutions": 1,
                        "deletions": 0,
                        "insertions": 0,
                        "ref_len": 100,
                        "hyp_len": 100,
                        "time_s": 0.1,
                    }
                )
    # one error row must be counted, not averaged
    rows.append(
        {
            "pair_id": "x-002", "lang": "ar", "condition": "shadow",
            "engine": "beta", "variant": "raw", "ok": False, "error": "boom",
            "cer": None, "wer": None, "substitutions": None, "deletions": None,
            "insertions": None, "ref_len": None, "hyp_len": None, "time_s": 0.0,
        }
    )
    return rows


def test_aggregate_pivot_math():
    agg = aggregate(_synthetic_rows())
    # alpha/auto mean over ar(0.05) + en(0.04) = 0.045
    assert agg["overall"]["alpha|auto"]["mean_cer"] == 0.045
    assert agg["overall"]["beta|raw"]["errors"] == 1
    assert agg["by_lang"]["alpha|auto|en"]["mean_cer"] == 0.04
    assert agg["by_condition"]["alpha|auto|clean"]["n"] == 2
    assert agg["error_counts"]["beta"] == 1
    # best variant per condition prefers auto over raw for both engines
    for engine in ("alpha", "beta"):
        best = agg["best_variant_per_condition"][f"{engine}|clean"]
        assert best["variant"] == "auto"


def test_to_markdown_contains_tables():
    agg = aggregate(_synthetic_rows())
    md = to_markdown(agg, {"engines": ["alpha", "beta"], "variants": ["raw", "auto"],
                            "n_pairs": 2, "normalization": "test"})
    assert "| engine | variant | n | mean CER | mean WER | errors |" in md
    assert "alpha" in md and "0.045" in md
    assert "Best variant per condition" in md


# ─── real-engine integration (self-skipping in CI) ───────────────────────────

_HAS_TESSERACT = shutil.which("tesseract") is not None


@pytest.mark.skipif(not _HAS_TESSERACT, reason="tesseract binary not on PATH")
def test_integration_tesseract_english_clean_page(tmp_path):
    """End-to-end smoke with the real engine — English only (CI-safe: no ara data needed)."""
    from ocr_core.benchmarks.golden.runner import GoldenTesseractEngine

    pair = next(p for p in golden_pairs() if p.lang == "en" and p.condition == "clean")
    eng = GoldenTesseractEngine(lang="eng", psm="6")
    result = run_golden_benchmark(
        engines=[eng], pairs=[pair], variants=("raw",), workdir=tmp_path
    )
    row = result["rows"][0]
    assert row["ok"], f"tesseract failed: {row['error']}"
    # a clean 36px page should OCR far better than chance
    assert row["cer"] < 0.30, f"unexpectedly bad CER on clean page: {row['cer']}"


@pytest.mark.skipif(
    not (_HAS_TESSERACT and os.environ.get("GOLDEN_TESSDATA_DIR")),
    reason="set GOLDEN_TESSDATA_DIR (dir containing ara.traineddata) to run",
)
def test_integration_tesseract_arabic_raqm_clean_page(tmp_path):
    """Arabic end-to-end: raqm + bundled Noto Naskh must produce near-perfect OCR.

    Proven 2026-10-05: bundled font + raqm → CER 0.0000 on clean pages with
    tessdata_fast ara. This test locks that quality floor in.
    """
    from ocr_core.benchmarks.golden.runner import GoldenTesseractEngine

    pair = next(p for p in golden_pairs() if p.lang == "ar" and p.condition == "clean")
    eng = GoldenTesseractEngine(
        lang="ara", psm="6", tessdata_dir=os.environ["GOLDEN_TESSDATA_DIR"]
    )
    result = run_golden_benchmark(
        engines=[eng], pairs=[pair], variants=("raw",), workdir=tmp_path
    )
    row = result["rows"][0]
    assert row["ok"], f"tesseract failed: {row['error']}"
    assert row["cer"] < 0.10, f"Arabic raqm path regressed: CER={row['cer']}"


def test_bundled_arabic_font_resolves():
    """The OFL Noto Naskh font ships inside the package for reproducibility."""
    from ocr_core.benchmarks.golden.render import resolve_font

    font = resolve_font("ar")
    assert font.is_file()
    assert font.name == "NotoNaskhArabic-Regular.ttf"
