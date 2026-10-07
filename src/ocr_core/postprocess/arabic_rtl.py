"""Arabic RTL post-processing — postprocess-layer entry point.

This module is the ``postprocess``-package home of the Arabic RTL fixer.
The implementation itself lives in :mod:`ocr_core.rtl_utils` (ported from
omni-medical-suite with documented adaptations); this wrapper adds the
postprocessor interface the unified pipeline consumes
(``fix(text) -> str``) and keeps two stable import paths:

- ``from ocr_core.rtl_utils import ArabicRTLFixer``                       (low-level)
- ``from ocr_core.postprocess.arabic_rtl import ArabicRTLPostProcessor``  (pipeline hook)

No new normalization rules are invented here — behavior is delegated
verbatim to ``rtl_utils`` so the two entry points can never drift apart.
"""
from __future__ import annotations

from ocr_core.rtl_utils import ArabicRTLFixer, RTLFixStats

__all__ = ["ArabicRTLFixer", "ArabicRTLPostProcessor", "RTLFixStats"]


class ArabicRTLPostProcessor:
    """Pipeline-facing wrapper around :class:`ocr_core.rtl_utils.ArabicRTLFixer`.

    Exposes the minimal hook used by ``ocr_core.pipeline.DocumentPipeline``
    (``fix``) plus stats/inspection passthroughs. Reuses a single underlying
    fixer instance — the fixer is stateless per call.
    """

    def __init__(self, reversal_threshold: float = 0.30) -> None:
        self._fixer = ArabicRTLFixer(reversal_threshold=reversal_threshold)

    def fix(self, text: str) -> str:
        """Return RTL-fixed text; non-Arabic input is returned unchanged."""
        return self._fixer.fix_text(text)

    def fix_with_stats(self, text: str, *, force: bool = False) -> tuple[str, RTLFixStats]:
        """RTL fix plus the underlying fixer's stats payload."""
        return self._fixer.analyze_and_fix(text, force=force)

    def should_fix(self, text: str) -> bool:
        return self._fixer.should_fix(text)

    @property
    def reversal_threshold(self) -> float:
        return self._fixer.reversal_threshold

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return f"ArabicRTLPostProcessor(reversal_threshold={self._fixer.reversal_threshold})"
