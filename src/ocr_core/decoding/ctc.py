"""
ctc.py
CTC (Connectionist Temporal Classification) decoding — from the published
definition only:

- Graves et al., "Connectionist Temporal Classification: Labelling
  Unsegmented Sequence Data with Recurrent Neural Networks", ICML 2006.
- Prefix beam search as popularized in A. Hannun et al.'s reference
  implementations (published algorithm, reimplemented here from its
  recurrence definition).

Input contract: a (T, C) matrix of per-timestep class probabilities —
T time frames, C classes including one blank. Rows that are not
normalized are softmaxed automatically (max-subtraction stable).

Why this matters for Arabic (docs/09 §2.5/§5): CTC removes the need for
pre-segmenting connected Arabic script into characters — the network
reads the whole line as a sequence. Decoding is what turns the frame
posteriors back into text; doing it in-package lets us attach
*per-character confidence* (docs/09 §5.2 item 3) for the human review
UI's amber checkmarks.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "ctc_greedy_decode",
    "ctc_greedy_decode_confident",
    "ctc_beam_search_decode",
]

FrameProbs = np.ndarray  # (T, C) float64


def _as_probabilities(mat: np.ndarray, blank_index: int) -> FrameProbs:
    """Validate a (T, C) score matrix and return normalized probabilities."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (T, C) matrix, got shape {arr.shape}")
    if arr.shape[0] == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError("probabilities contain NaN or infinity")
    if not (0 <= blank_index < arr.shape[1]):
        raise ValueError(
            f"blank_index {blank_index} out of range for {arr.shape[1]} classes"
        )

    row_sums = arr.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-3):
        norm = arr
    else:
        # treat input as logits → numerically stable softmax
        shifted = arr - arr.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        norm = exp / exp.sum(axis=1, keepdims=True)
    return np.clip(norm, 0.0, 1.0)


# ─── Greedy decoding ────────────────────────────────────────────────────────


def ctc_greedy_decode(
    mat: np.ndarray, alphabet: Sequence[str], blank_index: int = 0
) -> str:
    """Greedy CTC decode: per-frame argmax → collapse repeats → drop blanks.

    This is the fastest path (O(T)) and matches what production engines do
    when they only need the best path. `alphabet` maps class indices to
    tokens; the blank class decodes to nothing.
    """
    probs = _as_probabilities(mat, blank_index)
    if probs.shape[0] == 0:
        return ""
    if len(alphabet) != probs.shape[1]:
        raise ValueError(
            f"alphabet size {len(alphabet)} != class count {probs.shape[1]}"
        )

    best = probs.argmax(axis=1)
    out: List[str] = []
    prev = -1
    for cls in best:
        if cls != prev and cls != blank_index:
            out.append(str(alphabet[cls]))
        prev = cls
    return "".join(out)


def ctc_greedy_decode_confident(
    mat: np.ndarray, alphabet: Sequence[str], blank_index: int = 0
) -> List[Dict[str, object]]:
    """Greedy decode that also reports per-character confidence and frames.

    For each emitted character (a run of consecutive frames whose argmax
    is that class) the confidence is the mean posterior of the class over
    the run — a stable per-character score for review-UI checkmarks
    (docs/09 §5.2-3: per-char confidence → amber/verdict marks).

    Returns a list of dicts:
        {"char": str, "confidence": float in (0, 1],
         "start_frame": int, "end_frame": int}   # end exclusive
    ordered left-to-right in decoder time order (logical order; RTL
    reordering belongs to the text layer, not the decoder).
    """
    probs = _as_probabilities(mat, blank_index)
    if len(alphabet) != probs.shape[1]:
        raise ValueError(
            f"alphabet size {len(alphabet)} != class count {probs.shape[1]}"
        )
    result: List[Dict[str, object]] = []
    if probs.shape[0] == 0:
        return result

    best = probs.argmax(axis=1)
    t = 0
    T = probs.shape[0]
    while t < T:
        cls = int(best[t])
        run_start = t
        while t < T and int(best[t]) == cls:
            t += 1
        if cls != blank_index:
            conf = float(probs[run_start:t, cls].mean())
            result.append(
                {
                    "char": str(alphabet[cls]),
                    "confidence": conf,
                    "start_frame": run_start,
                    "end_frame": t,
                }
            )
    return result


# ─── Prefix beam search ─────────────────────────────────────────────────────


def ctc_beam_search_decode(
    mat: np.ndarray,
    alphabet: Sequence[str],
    blank_index: int = 0,
    beam_width: int = 8,
    prune_classes: int = 40,
) -> List[Tuple[str, float]]:
    """CTC prefix beam search (Graves 2006; published recurrence).

    Maintains, per prefix, the probability of paths ending in blank
    (p_b) and non-blank (p_nb), and extends prefixes per the standard
    recurrence each frame. Probabilities are renormalized per frame to
    avoid underflow on long sequences — scores remain comparable within
    one call's output list (they sum to ≈1 across returned prefixes).

    Args:
        mat: (T, C) probabilities or logits.
        alphabet: class-index → token mapping.
        blank_index: index of the blank class.
        beam_width: prefixes kept per frame.
        prune_classes: per frame, only the top-N non-blank classes by
            probability are expanded (blank always considered). Keeps the
            cost at O(T * beam * prune_classes) for wide alphabets.

    Returns up to `beam_width` (text, probability) pairs, best first.
    On unambiguous posteriors the top-1 equals the greedy decode; on
    ambiguous ones the beam can find a strictly higher-probability
    prefix than greedy (the classic case where beam search wins).
    """
    probs = _as_probabilities(mat, blank_index)
    if len(alphabet) != probs.shape[1]:
        raise ValueError(
            f"alphabet size {len(alphabet)} != class count {probs.shape[1]}"
        )
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if probs.shape[0] == 0:
        return [("", 1.0)]

    # beam: prefix tuple → [p_blank, p_nonblank]
    beam: Dict[Tuple[int, ...], List[float]] = {(): [1.0, 0.0]}

    for t in range(probs.shape[0]):
        frame = probs[t]
        # per-frame class pruning: blank + top prune_classes non-blank
        if prune_classes and probs.shape[1] - 1 > prune_classes:
            cand = np.argpartition(frame[1:], -prune_classes)[-prune_classes:] + 1
            classes = [blank_index] + [int(c) for c in cand]
        else:
            classes = list(range(probs.shape[1]))

        new_beam: Dict[Tuple[int, ...], List[float]] = {}

        def _add(prefix: Tuple[int, ...], pb: float, pnb: float) -> None:
            slot = new_beam.get(prefix)
            if slot is None:
                new_beam[prefix] = [pb, pnb]
            else:
                slot[0] += pb
                slot[1] += pnb

        for prefix, (p_b, p_nb) in beam.items():
            p_total = p_b + p_nb
            # extend with blank: stays on the same prefix, path may come
            # from either ending state
            _add(prefix, p_total * frame[blank_index], 0.0)
            last = prefix[-1] if prefix else None
            for c in classes:
                if c == blank_index:
                    continue
                p = frame[c]
                if p <= 0.0:
                    continue
                if c == last:
                    # same char again: only paths that just emitted a blank
                    # add a *new* occurrence (into the extended prefix);
                    # paths ending non-blank merge into the same prefix
                    _add(prefix, 0.0, p_nb * p)
                    _add(prefix + (c,), 0.0, p_b * p)
                else:
                    _add(prefix + (c,), 0.0, p_total * p)

        # prune to beam_width by total mass, then renormalize
        ranked = sorted(new_beam.items(), key=lambda kv: kv[1][0] + kv[1][1],
                        reverse=True)[:beam_width]
        total = sum(pb + pnb for _, (pb, pnb) in ranked)
        if total > 0.0:
            beam = {k: [pb / total, pnb / total] for k, (pb, pnb) in ranked}
        else:  # pragma: no cover — cannot happen with valid posteriors
            beam = {k: [pb, pnb] for k, (pb, pnb) in ranked}

    results = [
        ("".join(str(alphabet[c]) for c in prefix), p_b + p_nb)
        for prefix, (p_b, p_nb) in beam.items()
    ]
    results.sort(key=lambda item: item[1], reverse=True)
    return results
