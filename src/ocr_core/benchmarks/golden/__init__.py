"""Unified golden-sample benchmark (docs/09 §6 item 1).

Same pairs → same binarization variants → every available engine →
same normalization → one owned comparative CER table.
"""
from .dataset import CONDITIONS, MIN_PAIRS, GoldenPair, golden_pairs
from .render import render_pair
from .report import aggregate, save_outputs, to_markdown
from .runner import (
    VARIANTS,
    GoldenSuryaEngine,
    GoldenTesseractEngine,
    apply_variant,
    build_engines,
    run_golden_benchmark,
)

__all__ = [
    "CONDITIONS",
    "MIN_PAIRS",
    "GoldenPair",
    "golden_pairs",
    "render_pair",
    "aggregate",
    "save_outputs",
    "to_markdown",
    "VARIANTS",
    "GoldenSuryaEngine",
    "GoldenTesseractEngine",
    "apply_variant",
    "build_engines",
    "run_golden_benchmark",
]
