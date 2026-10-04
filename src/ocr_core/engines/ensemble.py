"""Ensemble engine — ported from omni-medical-suite ``src/ocr/ensemble.py``.

Adaptations to the ocr-core protocol (intentional, documented):
- omni auto-imported ``src.ocr.paddle_engine`` / ``src.ocr.easyocr_engine`` /
  ``src.layout.surya_analyzer`` and fed engines numpy arrays via
  ``extract_text()``. ocr-core engines are path-based
  (``process_image(path) -> OCRResult``), so engines are INJECTED at
  construction instead of auto-discovered, and the surya adapter is dropped
  (ocr-core does not ship a surya engine yet).
- Composite ranking kept from omni: confidence x length x validity_ratio,
  with the x0.1 penalty for engines that reported an error.
- base.py policy honored: a confidence of 0.0 means *unknown*. Engines in
  the unknown tier are ranked by a validity x length heuristic instead of
  fabricating a number. Engines with a real confidence (> 0.0) always
  outrank unknown-confidence engines.
- Per-engine failures are captured as error results — the ensemble never
  aborts because one engine raised.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

from ocr_core.engines.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)

_ARABIC_MIN, _ARABIC_MAX = 0x0600, 0x06FF


def _validity_ratio(text: str) -> float:
    """Share of letters/whitespace/Arabic-block characters in the text."""
    clean = text.strip()
    if not clean:
        return 0.0
    valid = sum(
        1
        for c in clean
        if c.isalpha() or c.isspace() or _ARABIC_MIN <= ord(c) <= _ARABIC_MAX
    )
    return valid / len(clean)


class EnsembleEngine(OCREngine):
    """Run several injected engines and pick the best OCRResult.

    Engines are passed as a ``{name: OCREngine}`` mapping (or added later
    via :meth:`register`). Ranking strategy:

    1. Engines whose ``OCRResult.confidence`` is a real value (> 0.0)
       compete by ``confidence * length * validity`` (omni composite).
    2. If no engine reports real confidence, the unknown tier competes by
       ``length * validity`` — a 0.0 confidence is never upgraded.
    3. Results with an error are penalized x0.1 (kept only if they
       still carry text).
    """

    name = "ensemble"
    ERROR_PENALTY = 0.1

    def __init__(self, engines: Optional[dict[str, OCREngine]] = None) -> None:
        self._engines: dict[str, OCREngine] = dict(engines or {})

    def register(self, name: str, engine: OCREngine) -> None:
        """Add (or replace) an engine in the ensemble."""
        self._engines[name] = engine

    @property
    def engine_names(self) -> list[str]:
        return list(self._engines)

    def available(self) -> bool:
        return any(engine.available() for engine in self._engines.values())

    def run_all(self, image_path) -> dict[str, OCRResult]:
        """Run every registered engine; per-engine failure is captured."""
        results: dict[str, OCRResult] = {}
        for name, engine in self._engines.items():
            if not engine.available():
                results[name] = OCRResult(
                    text="",
                    engine=name,
                    confidence=0.0,
                    error=f"engine '{name}' not available",
                )
                continue
            try:
                results[name] = engine.process_image(image_path)
            except Exception as e:  # fail-visible per engine, never abort
                logger.error("engine '%s' raised: %s", name, e)
                results[name] = OCRResult(
                    text="", engine=name, confidence=0.0, error=str(e)
                )
        return results

    # -- ranking -----------------------------------------------------------

    def _composite(self, result: OCRResult) -> float:
        clean_len = len(result.text.strip())
        validity = _validity_ratio(result.text)
        if result.confidence > 0.0:
            score = result.confidence * clean_len * validity
        else:
            # confidence unknown — heuristic, never a fabricated number
            score = clean_len * validity
        if result.error:
            score *= self.ERROR_PENALTY
        return score

    def process_image(self, image_path) -> OCRResult:
        started = time.perf_counter()
        results = self.run_all(image_path)

        if not self._engines:
            return OCRResult(
                text="",
                engine=self.name,
                confidence=0.0,
                error="no engines registered",
                processing_time=time.perf_counter() - started,
            )

        # eligible: clean success OR an error that still carried text
        eligible = {
            name: r
            for name, r in results.items()
            if (not r.error) or r.text.strip()
        }
        if not eligible:
            return OCRResult(
                text="",
                engine=self.name,
                confidence=0.0,
                error="all engines failed",
                processing_time=time.perf_counter() - started,
                meta={"per_engine": self._meta(results)},
            )

        confident = {n: r for n, r in eligible.items() if r.confidence > 0.0}
        pool = confident if confident else eligible
        strategy = "composite" if confident else "length-validity (confidence unknown)"

        best_name = max(pool, key=lambda n: (self._composite(pool[n]), n))
        best = pool[best_name]

        return OCRResult(
            text=best.text,
            engine=f"ensemble:{best_name}",
            confidence=best.confidence,  # 0.0 stays 0.0 — never fabricated
            processing_time=time.perf_counter() - started,
            meta={
                "strategy": strategy,
                "winner": best_name,
                "per_engine": self._meta(results),
            },
        )

    def process_pdf(self, pdf_path, max_pages: Optional[int] = None) -> OCRResult:
        # The omni ensemble was image-only; PDF handling stays with the
        # per-engine implementations. Fail visibly instead of guessing.
        return OCRResult(
            text="",
            engine=self.name,
            confidence=0.0,
            error="EnsembleEngine.process_pdf not supported — route PDFs to a concrete engine",
        )

    # -- reporting ---------------------------------------------------------

    def comparison_table(self, image_path) -> str:
        """Formatted per-engine comparison (ported from omni)."""
        results = self.run_all(image_path)
        separator = "=" * 72
        lines = [
            separator,
            "OCR Engine Comparison Table (ensemble)",
            separator,
            f"{'Engine':<16} {'Chars':>8} {'Conf':>8} {'Status':>8}",
            "-" * 72,
        ]
        for name, r in results.items():
            chars = len(r.text.strip())
            conf = f"{r.confidence:.3f}" if r.confidence > 0.0 else "N/A"
            status = "ok" if r.ok else "ERR"
            lines.append(f"{name:<16} {chars:>8} {conf:>8} {status:>8}")
        lines.append("-" * 72)
        lines.append(f"engines used: {sum(1 for r in results.values() if r.ok)}")
        lines.append(separator)
        return "\n".join(lines)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _meta(results: dict[str, OCRResult]) -> dict[str, dict]:
        return {
            name: {
                "confidence": r.confidence,
                "error": r.error,
                "chars": len(r.text.strip()),
            }
            for name, r in results.items()
        }
