"""Smart OCR engine routing — ported from omni-medical-suite
``packages/core/engine_router.py`` (MIT).

Deviations from the omni original (intentional, documented):
- ``from app.core.decision_log import log_decision`` replaced by
  ``ocr_core.telemetry.log_decision`` (same signature, stdlib-logging sink).
- ``EngineRegistry`` coupling is duck-typed: any object exposing
  ``_probed`` and ``available_engine_names()`` may be passed; the omni
  ``engine_registry`` module itself was NOT ported (its TrOCR/Qwen/QARI/
  Nougat adapters probe engines ocr-core does not ship).
- Engine name constants and profiles are kept verbatim so recommendations
  stay comparable with omni. Engines ocr-core does not ship (TrOCR, QARI,
  Nougat, Qwen) remain valid *recommendations* — they are advisory strings;
  the caller decides what to instantiate.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ocr_core.telemetry import log_decision

logger = logging.getLogger(__name__)

ENGINE_EASYOCR = "EasyOCR"
ENGINE_TROCR = "TrOCR"
ENGINE_TESSERACT = "Tesseract"
ENGINE_PADDLE = "PaddleOCR"
ENGINE_QWEN_HANDWRITTEN = "Arabic-handwritten-OCR (Qwen)"
ENGINE_QARI = "QARI"
ENGINE_NOUGAT = "Nougat"

ENGINE_RAM_REQUIREMENTS = {
    ENGINE_TESSERACT: 0.5,
    ENGINE_EASYOCR: 1.5,
    ENGINE_TROCR: 3.5,
    ENGINE_PADDLE: 4.0,
    ENGINE_QWEN_HANDWRITTEN: 5.5,
    ENGINE_QARI: 4.5,
    ENGINE_NOUGAT: 4.5,
}

PROFILE_ENGINES = {
    "low": [ENGINE_TESSERACT, ENGINE_EASYOCR],
    "balanced": [
        ENGINE_EASYOCR,
        ENGINE_TESSERACT,
        ENGINE_TROCR,
        ENGINE_QWEN_HANDWRITTEN,
        ENGINE_QARI,
    ],
    "high": [
        ENGINE_EASYOCR,
        ENGINE_TESSERACT,
        ENGINE_TROCR,
        ENGINE_PADDLE,
        ENGINE_QWEN_HANDWRITTEN,
        ENGINE_QARI,
        ENGINE_NOUGAT,
    ],
}


class EngineRouter:
    """Select the most appropriate OCR engines without loading everything.

    Can optionally use a registry-like object to filter recommendations
    by *actual runtime availability* rather than static profile strings.
    """

    def __init__(
        self,
        profile: str = "balanced",
        use_gpu: bool = False,
        max_engines: int = 2,
        available_ram_gb: float = 8.0,
        registry: Optional[Any] = None,
    ) -> None:
        self.profile = profile
        self.use_gpu = use_gpu
        self.max_engines = max_engines
        self.available_ram_gb = available_ram_gb
        self._registry = registry
        self._allowed = list(PROFILE_ENGINES.get(profile, PROFILE_ENGINES["balanced"]))

        # If a registry is provided, intersect allowed engines with actually
        # available ones (duck-typed: _probed + available_engine_names()).
        if registry and getattr(registry, "_probed", False):
            available_names = set(registry.available_engine_names())
            self._allowed = [e for e in self._allowed if e in available_names]
            if not self._allowed:
                self._allowed = [ENGINE_TESSERACT]
                logger.warning(
                    "No engines from profile passed availability check — "
                    "falling back to Tesseract"
                )

        logger.info(
            "EngineRouter init: profile=%s gpu=%s max=%d allowed=%s",
            profile, use_gpu, max_engines, self._allowed,
        )

    def select(
        self,
        image_quality: float = 0.80,
        language: str = "ar",
        block_type: str = "paragraph",
        *,
        has_diacritics: bool = False,
        document_type: str = "generic",
        prefer_structured_output: bool = False,
    ) -> tuple[list[str], list[str]]:
        import time as _time

        t0 = _time.perf_counter()
        recommendations: list[str] = []
        reasons: list[str] = []
        # Track which candidates were considered but skipped (for the
        # structured decision log). Reset per call.
        skipped: list[str] = []

        if self.profile == "low":
            fallback = [
                engine
                for engine in [ENGINE_TESSERACT, ENGINE_EASYOCR]
                if engine in self._allowed
            ]
            chosen = fallback[:1]
            chosen_reasons = ["low-end profile — single engine mode"]
            log_decision(
                decision="engine_selection",
                outcome=chosen,
                reasons=chosen_reasons,
                inputs={
                    "profile": self.profile,
                    "language": language,
                    "image_quality": image_quality,
                    "block_type": block_type,
                    "has_diacritics": has_diacritics,
                    "document_type": document_type,
                    "prefer_structured_output": prefer_structured_output,
                    "allowed": self._allowed,
                },
                skipped=[e for e in self._allowed if e not in chosen],
                duration_ms=(_time.perf_counter() - t0) * 1000,
            )
            return chosen, chosen_reasons

        if block_type == "handwriting":
            for engine, reason in [
                (ENGINE_QWEN_HANDWRITTEN, "handwriting detected — Qwen handwritten OCR first"),
                (ENGINE_QARI, "handwriting fallback for Arabic forms"),
                (ENGINE_TROCR, "legacy handwriting fallback"),
            ]:
                if engine in self._allowed and engine not in recommendations:
                    recommendations.append(engine)
                    reasons.append(reason)
                elif engine not in self._allowed:
                    skipped.append(engine)
                if len(recommendations) >= self.max_engines:
                    break

        if has_diacritics and language in {"ar", "mixed"} and ENGINE_QARI in self._allowed:
            if ENGINE_QARI not in recommendations:
                recommendations.append(ENGINE_QARI)
                reasons.append("vocalized Arabic / tashkeel detected")

        if (
            prefer_structured_output
            or block_type in {"table", "form"}
            or document_type in {"report", "article", "book", "markdown"}
        ) and ENGINE_NOUGAT in self._allowed:
            if ENGINE_NOUGAT not in recommendations:
                recommendations.append(ENGINE_NOUGAT)
                reasons.append("structure-preserving extraction requested")

        if language in {"ar", "mixed"} and ENGINE_EASYOCR in self._allowed:
            if ENGINE_EASYOCR not in recommendations:
                recommendations.append(ENGINE_EASYOCR)
                reasons.append(f"Arabic/mixed language ({language})")

        if (
            image_quality < 0.60
            and ENGINE_TESSERACT in self._allowed
            and ENGINE_TESSERACT not in recommendations
        ):
            recommendations.append(ENGINE_TESSERACT)
            reasons.append("low image quality — conservative fallback")

        if (
            self.profile == "high"
            and language in {"ar", "mixed"}
            and ENGINE_PADDLE in self._allowed
        ):
            if ENGINE_PADDLE not in recommendations:
                recommendations.append(ENGINE_PADDLE)
                reasons.append("high profile — PaddleOCR fallback for printed Arabic")

        if language in {"en", "de"} and image_quality >= 0.75:
            for engine, reason in [
                (ENGINE_TROCR, f"Latin script + high quality ({language})"),
                (ENGINE_EASYOCR, f"Latin fallback ({language})"),
            ]:
                if engine in self._allowed and engine not in recommendations:
                    recommendations.append(engine)
                    reasons.append(reason)

        if not recommendations:
            fallback = [
                engine
                for engine in [ENGINE_EASYOCR, ENGINE_TESSERACT]
                if engine in self._allowed
            ]
            recommendations = fallback[: self.max_engines]
            reasons = ["default fallback — no specific signal"] * len(recommendations)

        pre_ram_recommendations = list(recommendations)
        pre_ram_reasons = list(reasons)
        recommendations, reasons = self._filter_by_ram(recommendations, reasons)
        # Anything dropped by the RAM filter is recorded as skipped
        for engine in pre_ram_recommendations:
            if engine not in recommendations:
                skipped.append(engine)

        seen: set[str] = set()
        deduped_engines: list[str] = []
        deduped_reasons: list[str] = []
        for engine, reason in zip(recommendations, reasons, strict=False):
            if engine not in seen:
                seen.add(engine)
                deduped_engines.append(engine)
                deduped_reasons.append(reason)
            if len(deduped_engines) >= self.max_engines:
                break
        recommendations = deduped_engines
        reasons = deduped_reasons

        # Record engines considered but not selected (informational)
        for engine in self._allowed:
            if engine not in recommendations and engine not in skipped:
                skipped.append(engine)

        log_decision(
            decision="engine_selection",
            outcome=recommendations,
            reasons=reasons,
            inputs={
                "profile": self.profile,
                "language": language,
                "image_quality": image_quality,
                "block_type": block_type,
                "has_diacritics": has_diacritics,
                "document_type": document_type,
                "prefer_structured_output": prefer_structured_output,
                "allowed": self._allowed,
                "max_engines": self.max_engines,
                "available_ram_gb": self.available_ram_gb,
            },
            skipped=skipped,
            duration_ms=(_time.perf_counter() - t0) * 1000,
        )
        return recommendations, reasons

    def estimate_time(self, engines: list[str]) -> float:
        estimates = {
            ENGINE_TESSERACT: 0.4,
            ENGINE_EASYOCR: 1.5,
            ENGINE_TROCR: 2.5,
            ENGINE_PADDLE: 2.0,
            ENGINE_QWEN_HANDWRITTEN: 3.4,
            ENGINE_QARI: 2.8,
            ENGINE_NOUGAT: 3.2,
        }
        total = sum(estimates.get(engine, 1.0) for engine in engines)
        if not self.use_gpu:
            total *= 1.8
        return round(total, 1)

    def summary(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "use_gpu": self.use_gpu,
            "max_engines": self.max_engines,
            "available_ram_gb": self.available_ram_gb,
            "allowed_engines": self._allowed,
        }

    def _filter_by_ram(
        self,
        engines: list[str],
        reasons: list[str],
    ) -> tuple[list[str], list[str]]:
        total_ram = 0.0
        filtered_engines: list[str] = []
        filtered_reasons: list[str] = []
        for engine, reason in zip(engines, reasons, strict=False):
            requirement = ENGINE_RAM_REQUIREMENTS.get(engine, 1.0)
            if total_ram + requirement <= self.available_ram_gb:
                filtered_engines.append(engine)
                filtered_reasons.append(reason)
                total_ram += requirement
            else:
                logger.warning(
                    "Skipping %s (needs %.1fGB, available %.1fGB)",
                    engine,
                    requirement,
                    self.available_ram_gb - total_ram,
                )
        if filtered_engines:
            return filtered_engines, filtered_reasons
        return [ENGINE_TESSERACT], ["RAM-constrained fallback"]

    @classmethod
    def from_config(cls, config) -> "EngineRouter":
        try:
            import psutil

            ram_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            ram_gb = 8.0
        return cls(
            profile=getattr(config, "engine_profile", "balanced"),
            use_gpu=getattr(config, "use_gpu", False),
            max_engines=getattr(config, "router_max_engines", 2),
            available_ram_gb=ram_gb,
        )
