"""
Tests for ocr_core.decoding.ctc — CTC greedy/prefix-beam decoding
(Graves et al., ICML 2006) with per-character confidence.

All fixtures are tiny hand-computed probability matrices so the expected
decodes are derivable by inspection.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_core.decoding.ctc import (
    ctc_beam_search_decode,
    ctc_greedy_decode,
    ctc_greedy_decode_confident,
)

ALPHABET = ["_", "a", "b"]  # index 0 = blank


def _peaked(choices, num_classes=3):
    """Build a (T, C) matrix where each frame is a near-one-hot row.

    choices: list of (class_index, peak_value) per frame; the remainder of
    the row's mass is spread on the first non-chosen class to keep rows
    valid probability distributions.
    """
    rows = []
    for cls, peak in choices:
        row = np.zeros(num_classes)
        row[cls] = peak
        leftover = 1.0 - peak
        fill = 1 if cls != 1 else 2
        row[fill] = leftover
        rows.append(row)
    return np.asarray(rows)


# ─── Greedy ──────────────────────────────────────────────────────────────────


def test_greedy_basic_sequence():
    probs = _peaked([(1, 0.95), (0, 0.95), (2, 0.95)])  # a _ b
    assert ctc_greedy_decode(probs, ALPHABET) == "ab"


def test_greedy_collapses_consecutive_repeats():
    probs = _peaked([(1, 0.9), (1, 0.9), (1, 0.9)])  # a a a
    assert ctc_greedy_decode(probs, ALPHABET) == "a"


def test_greedy_blank_separates_repeats():
    probs = _peaked([(1, 0.9), (0, 0.9), (1, 0.9)])  # a _ a
    assert ctc_greedy_decode(probs, ALPHABET) == "aa"


def test_greedy_empty_and_blank_only_inputs():
    assert ctc_greedy_decode(np.zeros((0, 3)), ALPHABET) == ""
    probs = _peaked([(0, 0.99), (0, 0.99)])
    assert ctc_greedy_decode(probs, ALPHABET) == ""


def test_greedy_accepts_logits_via_softmax():
    normalized = _peaked([(1, 0.95), (2, 0.95)])
    logits = np.log(np.clip(normalized, 1e-9, 1.0)) + 7.0  # finite, unnormalized
    assert ctc_greedy_decode(logits, ALPHABET) == "ab"


def test_validation_errors():
    with pytest.raises(ValueError):
        ctc_greedy_decode(np.zeros((3, 4)), ALPHABET)          # C mismatch
    with pytest.raises(ValueError):
        ctc_greedy_decode(np.zeros(5), ALPHABET)               # not 2D
    with pytest.raises(ValueError):
        ctc_greedy_decode(np.full((2, 3), np.nan), ALPHABET)   # NaN
    with pytest.raises(ValueError):
        ctc_greedy_decode(_peaked([(1, 0.9)]), ALPHABET, blank_index=9)


# ─── Per-character confidence ────────────────────────────────────────────────


def test_confident_fields_and_order():
    probs = _peaked([(1, 0.95), (1, 0.85), (0, 0.9), (2, 0.7)])  # aa _ b
    hyp = ctc_greedy_decode_confident(probs, ALPHABET)
    assert [h["char"] for h in hyp] == ["a", "b"]
    assert hyp[0]["start_frame"] == 0 and hyp[0]["end_frame"] == 2
    assert hyp[1]["start_frame"] == 3 and hyp[1]["end_frame"] == 4
    for h in hyp:
        assert 0.0 < h["confidence"] <= 1.0


def test_confident_mean_over_run():
    probs = _peaked([(1, 0.9), (1, 0.7)])  # two frames of 'a'
    hyp = ctc_greedy_decode_confident(probs, ALPHABET)
    assert len(hyp) == 1
    assert hyp[0]["confidence"] == pytest.approx(0.8)


def test_confident_skips_blank_runs():
    probs = _peaked([(0, 0.99), (1, 0.8), (0, 0.99)])
    hyp = ctc_greedy_decode_confident(probs, ALPHABET)
    assert [h["char"] for h in hyp] == ["a"]
    assert hyp[0]["start_frame"] == 1 and hyp[0]["end_frame"] == 2


def test_confident_empty_input():
    assert ctc_greedy_decode_confident(np.zeros((0, 3)), ALPHABET) == []


# ─── Prefix beam search ──────────────────────────────────────────────────────


def test_beam_top1_matches_greedy_on_sharp_distribution():
    probs = _peaked([(1, 0.99), (0, 0.99), (2, 0.99)])
    beam = ctc_beam_search_decode(probs, ALPHABET, beam_width=5)
    assert beam[0][0] == ctc_greedy_decode(probs, ALPHABET) == "ab"
    assert beam[0][1] > 0.9


def test_beam_finds_higher_probability_prefix_than_greedy():
    """The classic case: the greedy path 'a→b' has P=0.25, but the prefix
    'a' accumulates P=0.32 (a_, aa, _a) — beam search must return 'a'
    first, which greedy per-frame argmax cannot see."""
    frame1 = np.array([0.20, 0.50, 0.30])  # blank a b
    frame2 = np.array([0.15, 0.35, 0.50])
    probs = np.stack([frame1, frame2])
    greedy = ctc_greedy_decode(probs, ALPHABET)
    beam = dict(ctc_beam_search_decode(probs, ALPHABET, beam_width=10))
    assert greedy == "ab"                       # per-frame argmaxes
    assert beam["ab"] == pytest.approx(0.25, abs=1e-9)
    assert beam["a"] == pytest.approx(0.32, abs=1e-9)
    assert beam["a"] > beam["ab"]               # the higher-probability prefix
    assert max(beam, key=beam.get) == "a"


def test_beam_scores_approx_normalized():
    probs = _peaked([(1, 0.9), (2, 0.8), (1, 0.7)])
    beam = ctc_beam_search_decode(probs, ALPHABET, beam_width=8)
    assert sum(p for _, p in beam) == pytest.approx(1.0, abs=1e-6)
    probs_sorted = [p for _, p in beam]
    assert probs_sorted == sorted(probs_sorted, reverse=True)


def test_beam_width_limits_results():
    probs = _peaked([(1, 0.9), (2, 0.9)])
    beam = ctc_beam_search_decode(probs, ALPHABET, beam_width=1)
    assert len(beam) == 1
    with pytest.raises(ValueError):
        ctc_beam_search_decode(probs, ALPHABET, beam_width=0)


def test_beam_empty_input():
    assert ctc_beam_search_decode(np.zeros((0, 3)), ALPHABET) == [("", 1.0)]


def test_beam_repeated_char_needs_blank_between():
    """a a without an intervening blank must decode to 'a' (same-class
    extension rule); a _ a decodes to 'aa'."""
    probs = _peaked([(1, 0.99), (1, 0.99)])
    beam = dict(ctc_beam_search_decode(probs, ALPHABET, beam_width=4))
    assert max(beam, key=beam.get) == "a"

    probs = _peaked([(1, 0.99), (0, 0.99), (1, 0.99)])
    beam = dict(ctc_beam_search_decode(probs, ALPHABET, beam_width=4))
    assert max(beam, key=beam.get) == "aa"


def test_beam_class_pruning_keeps_mass():
    """With a wide alphabet, pruning to top-k classes must not change the
    decode when the true classes dominate."""
    rng_probs = np.full((5, 60), 1e-4)
    seq = [10, 0, 20, 0, 20]
    for t, c in enumerate(seq):
        rng_probs[t] = 1e-4
        rng_probs[t, c] = 0.999
    alphabet = ["_"] + [chr(96 + i) for i in range(1, 60)]
    # j _ t _ t  →  'jtt'
    assert ctc_beam_search_decode(rng_probs, alphabet, beam_width=8,
                                  prune_classes=5)[0][0] == "jtt"
