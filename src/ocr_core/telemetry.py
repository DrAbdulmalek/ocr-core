"""Decision telemetry for ocr-core.

Replaces omni-medical-suite's ``app.core.decision_log``: same call
signature (decision / outcome / reasons / inputs / skipped / duration_ms)
but the sink is structured stdlib logging instead of the omni database.
Attach a handler to the ``ocr_core.telemetry`` logger to capture decisions.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ocr_core.telemetry")


def log_decision(
    decision: str,
    outcome: Any,
    reasons: list[str],
    inputs: dict[str, Any] | None = None,
    skipped: list[str] | None = None,
    duration_ms: float = 0.0,
) -> None:
    """Record one routing/selection decision in a structured, greppable line."""
    logger.info(
        "decision=%s outcome=%s reasons=%s skipped=%s inputs=%s duration_ms=%.2f",
        decision,
        outcome,
        reasons,
        skipped or [],
        inputs or {},
        duration_ms,
    )
