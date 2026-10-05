"""Sequence decoding for CTC-trained recognizers (LSTM/CRNN engines).

Implements the published decoding algorithms of Connectionist Temporal
Classification (Graves et al., ICML 2006) on top of NumPy arrays, so they
can be applied to any engine that exposes per-timestep class probabilities
(Tesseract 5's LSTM lines, CRNN heads in PaddleOCR/EasyOCR, or our own
models) without pulling in a deep-learning runtime.
"""

from .ctc import (
    ctc_beam_search_decode,
    ctc_greedy_decode,
    ctc_greedy_decode_confident,
)

__all__ = [
    "ctc_greedy_decode",
    "ctc_greedy_decode_confident",
    "ctc_beam_search_decode",
]
