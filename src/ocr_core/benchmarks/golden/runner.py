"""Unified golden-sample runner — same pairs, same normalization, N engines.

docs/09 §6 item 1 contract:
  * every golden pair is rendered once,
  * each page goes through the SAME binarization variants
    (raw / otsu / sauvola / auto — auto is PR #14's router),
  * each variant image is fed to EVERY available engine
    (tesseract ara+eng, surya when installed, or injected stubs in tests),
  * OCR output and ground truth are scored after the SAME normalization
    (``arabic_strong_normalize`` applied to both sides — apples to apples),
  * CER/WER come from the package's own ``EditDistance`` (S/D/I breakdown).

Engines never raise: failures land in ``rows`` as ``error`` entries, so one
broken page cannot silently kill a benchmark run (package policy).
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from ocr_core.benchmarks.core.metrics import EditDistance
from ocr_core.engines.base import OCRResult
from ocr_core.postprocess.normalization import arabic_strong_normalize
from ocr_core.preprocess.binarize import (
    binarize_auto,
    binarize_otsu,
    binarize_sauvola,
)

from .dataset import GoldenPair, golden_pairs
from .render import render_pair

__all__ = [
    "VARIANTS",
    "GoldenTesseractEngine",
    "GoldenSuryaEngine",
    "build_engines",
    "apply_variant",
    "run_golden_benchmark",
]

#: binarization variants compared by the benchmark
VARIANTS = ("raw", "otsu", "sauvola", "auto")


def apply_variant(gray: np.ndarray, variant: str) -> np.ndarray:
    """Apply one binarization variant to a 2-D grayscale page."""
    if variant == "raw":
        return gray
    if variant == "otsu":
        return binarize_otsu(gray)
    if variant == "sauvola":
        return binarize_sauvola(gray)
    if variant == "auto":
        return binarize_auto(gray)["image"]  # type: ignore[index]
    raise ValueError(f"unknown variant: {variant!r} (expected one of {VARIANTS})")


class GoldenTesseractEngine:
    """Tesseract adapter with explicit ``--tessdata-dir`` support.

    Wraps :class:`ocr_core.engines.tesseract.TesseractEngine` semantics but
    allows pointing at the documented ``tessdata_fast`` models (PR #12
    provenance) without mutating the shared engine's fixed config format.
    """

    name = "tesseract"

    def __init__(self, lang: str = "ara+eng", psm: str = "6",
                 tessdata_dir: Optional[str] = None, dpi: int = 300):
        self.lang = lang
        self.psm = psm
        self.tessdata_dir = str(tessdata_dir) if tessdata_dir else None
        self.dpi = dpi

    def available(self) -> bool:
        return shutil.which("tesseract") is not None

    def process_image(self, image_path) -> OCRResult:
        import pytesseract  # lazy: optional extra

        start = time.perf_counter()
        config = f"--psm {self.psm} --dpi {self.dpi}"
        if self.tessdata_dir:
            config += f" --tessdata-dir {self.tessdata_dir}"
        try:
            text = pytesseract.image_to_string(
                Image.open(image_path), lang=self.lang, config=config
            )
        except Exception as exc:  # noqa: BLE001 - engine failures are reported
            return OCRResult(engine=self.name, error=f"{type(exc).__name__}: {exc}")
        return OCRResult(
            text=text.strip(),
            engine=self.name,
            confidence=0.0,  # never invented
            processing_time=time.perf_counter() - start,
            meta={"lang": self.lang, "psm": self.psm,
                  "tessdata_dir": self.tessdata_dir or "system"},
        )


class GoldenSuryaEngine:
    """Best-effort Surya adapter (optional extra: ``pip install .[surya]``).

    The upstream API moved several times across 0.x releases, so construction
    and calls are deliberately defensive: any failure becomes an
    ``OCRResult.error`` row, never an exception escaping the benchmark.
    """

    name = "surya"

    def __init__(self, languages: Sequence[str] = ("ar", "en")):
        from surya.ocr import OCR  # raises ImportError → engine unavailable

        self.languages = list(languages)
        self._ocr = OCR(languages=self.languages)

    def available(self) -> bool:
        return True

    def process_image(self, image_path) -> OCRResult:
        start = time.perf_counter()
        try:
            from PIL import Image as _Image

            img = _Image.open(image_path)
            try:
                preds = self._ocr.run(img)
            except TypeError:
                # some versions: run([img], [langs]) or run(img, langs)
                try:
                    preds = self._ocr.run([img], [self.languages])
                except TypeError:
                    preds = self._ocr.run([img])
            texts: list[str] = []
            lines = getattr(preds, "text_lines", preds)
            for p in lines:
                texts.append(getattr(p, "text", str(p)))
            return OCRResult(
                text="\n".join(texts).strip(),
                engine=self.name,
                confidence=0.0,  # never invented
                processing_time=time.perf_counter() - start,
                meta={"languages": self.languages},
            )
        except Exception as exc:  # noqa: BLE001 - engine failures are reported
            return OCRResult(engine=self.name, error=f"{type(exc).__name__}: {exc}")


def build_engines(
    spec: Iterable[str] = ("tesseract", "surya"),
    tessdata_dir: Optional[str] = None,
) -> tuple[list, list[str]]:
    """Instantiate requested engines; report (not raise) unavailability.

    Returns ``(engines, notes)`` — *notes* lists human-readable availability
    decisions so the benchmark report can honestly state what ran.
    """
    engines: list = []
    notes: list[str] = []
    for item in spec:
        item = item.strip().lower()
        if item == "tesseract":
            eng = GoldenTesseractEngine(tessdata_dir=tessdata_dir)
            if eng.available():
                engines.append(eng)
                notes.append("tesseract: available (binary on PATH)")
            else:
                notes.append("tesseract: SKIPPED — binary not on PATH")
        elif item == "surya":
            try:
                engines.append(GoldenSuryaEngine())
                notes.append("surya: available (import ok)")
            except ImportError:
                notes.append(
                    "surya: SKIPPED — surya-ocr not installed "
                    "(pip install .[surya]); runner continues without it"
                )
            except Exception as exc:  # noqa: BLE001 - init failure is a note
                notes.append(f"surya: SKIPPED — init failed: {exc}")
        else:
            notes.append(f"{item}: SKIPPED — unknown engine spec")
    return engines, notes


def run_golden_benchmark(
    engines: Sequence,
    variants: Sequence[str] = VARIANTS,
    pairs: Optional[Sequence[GoldenPair]] = None,
    workdir: Optional[Path] = None,
    use_medical_dict: bool = True,
    limit: Optional[int] = None,
) -> dict:
    """Run the full golden matrix and return ``{"meta", "rows"}``.

    ``engines`` is any sequence of objects exposing ``.name`` and
    ``.process_image(path) -> OCRResult`` (real adapters or test stubs).
    """
    if not engines:
        raise ValueError("no engines provided — nothing to benchmark")
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant: {v!r}")

    pairs = list(pairs) if pairs is not None else golden_pairs()
    if limit is not None:
        pairs = pairs[:limit]

    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="golden-"))
    workdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pair in pairs:
        page = render_pair(pair)
        base_path = workdir / f"{pair.id}.png"
        page.save(base_path, format="PNG")

        base_gray = np.asarray(page, dtype=np.uint8)  # already mode "L"
        for variant in variants:
            variant_arr = apply_variant(base_gray, variant)
            vpath = workdir / f"{pair.id}.{variant}.png"
            Image.fromarray(variant_arr, mode="L").save(vpath, format="PNG")

            ref_norm = arabic_strong_normalize(
                pair.text, use_medical_dict=use_medical_dict
            )
            for engine in engines:
                result = engine.process_image(vpath)
                elapsed = result.processing_time or 0.0
                if not result.ok:
                    rows.append(
                        {
                            "pair_id": pair.id,
                            "lang": pair.lang,
                            "condition": pair.condition,
                            "engine": engine.name,
                            "variant": variant,
                            "ok": False,
                            "error": result.error,
                            "cer": None,
                            "wer": None,
                            "substitutions": None,
                            "deletions": None,
                            "insertions": None,
                            "ref_len": len(ref_norm),
                            "hyp_len": None,
                            "time_s": round(elapsed, 4),
                        }
                    )
                    continue
                hyp_norm = arabic_strong_normalize(
                    result.text, use_medical_dict=use_medical_dict
                )
                cer = EditDistance.cer(ref_norm, hyp_norm)
                wer = EditDistance.wer(ref_norm, hyp_norm)
                rows.append(
                    {
                        "pair_id": pair.id,
                        "lang": pair.lang,
                        "condition": pair.condition,
                        "engine": engine.name,
                        "variant": variant,
                        "ok": True,
                        "error": None,
                        "cer": round(cer["cer"], 6),
                        "wer": round(wer["wer"], 6),
                        "substitutions": cer["substitutions"],
                        "deletions": cer["deletions"],
                        "insertions": cer["insertions"],
                        "ref_len": cer["ref_length"],
                        "hyp_len": len(hyp_norm),
                        "time_s": round(elapsed, 4),
                    }
                )

    return {
        "meta": {
            "variants": list(variants),
            "engines": [e.name for e in engines],
            "n_pairs": len(pairs),
            "use_medical_dict": use_medical_dict,
            "normalization": "arabic_strong_normalize (both sides)",
            "workdir": str(workdir),
        },
        "rows": rows,
    }
