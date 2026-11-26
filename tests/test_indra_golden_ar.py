"""Golden Arabic smoke-eval for the optional Indra adapter — v1 (synthetic render).

Runs ONLY when indra-ocr is installed AND INDRA_GOLDEN=1 (first run downloads
~76 MB of ONNX weights from Hugging Face; CPU inference only).

Purpose: a reproducible CEILING check on rendered, clean Arabic text that keeps
text quality (CER / WER) separate from layout quality (line recall + box IoU).
It is NOT a production-readiness benchmark: real scanned medical documents are
a separate, still-open gate.

Golden set: indra-ar-golden-r1 (2026-10-23) — 8 rendered lines, 1 line per image.
Eval protocol:
  - Arabic-Indic digits in hypotheses are normalized to Latin digits.
  - CER = Levenshtein(ref, hyp) / len(ref) on the raw strings (spaces kept).
  - WER = Levenshtein on whitespace tokens / len(ref tokens).
  - IoU between each reference render bbox and the best-matching engine box.
  - Gates encode the r1 BASELINE (regression tripwire), not a quality bar:
    mean CER <= 0.06, mean WER <= 0.30, recall >= 0.95, mean IoU >= 0.85,
    exact-match lines >= 6/8.
Known v1 deviations recorded as baseline (NOT normalized away):
  - Date components may be re-ordered by the engine: 2026-10-23 -> 23-10-2026
    (RTL date-order behaviour — a REAL hazard for medical date fields).
  - علي/على alif-variant confusion inside names.
"""
import json
import os

import pytest

indra_ocr = pytest.importorskip("indra_ocr")

pytestmark = pytest.mark.skipif(
    os.environ.get("INDRA_GOLDEN") != "1",
    reason="golden eval needs indra-ocr weights; set INDRA_GOLDEN=1 to run",
)

GOLDEN_VERSION = "indra-ar-golden-r1"
GOLDEN_LINES = [
    "مستشفى الملك فهد",
    "الفحص: تحليل الدم",
    "اسم المريض: أحمد بن علي",
    "الجرعة: 500 ملغ كل 8 ساعات",
    "ضغط الدم 120 على 80",
    "تاريخ الفحص 2026-10-23",
    "نتيجة التحليل سلبي",
    "يكرر المراجعة بعد أسبوعين",
]

_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize(s: str) -> str:
    return (s or "").translate(_DIGIT_MAP).strip()


def _lev(a: str, b: str) -> int:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b)


def _cer(ref: str, hyp: str) -> float:
    return _lev(ref, hyp) / max(len(ref), 1)


def _wer(ref: str, hyp: str) -> float:
    rt, ht = ref.split(), hyp.split()
    return _lev(" ".join(rt), " ".join(ht)) / max(len(rt), 1)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _render_all(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    font_path = None
    for cand in [
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(cand):
            font_path = cand
            break
    assert font_path, "no Arabic-capable font found for golden render"
    font = ImageFont.truetype(font_path, 44)

    refs = []
    for i, line in enumerate(GOLDEN_LINES):
        img = Image.new("RGB", (1600, 160), "white")
        d = ImageDraw.Draw(img)
        bbox = d.textbbox((80, 40), line, font=font, direction="rtl", language="ar")
        d.text((80, 40), line, font=font, fill="black", direction="rtl", language="ar")
        p = tmp_path / f"golden_ar_{i:02d}.png"
        img.save(p)
        refs.append({"path": str(p), "ref": line, "bbox": list(bbox)})
    return refs


def test_indra_arabic_golden_r1(tmp_path):
    from ocr_core.engines.indra import IndraEngine

    engine = IndraEngine(threads=4)
    refs = _render_all(tmp_path)

    per_line = []
    for r in refs:
        res = engine.process_image(r["path"])
        assert res.ok, f"engine error on {r['path']}: {res.error}"
        hyp = _normalize(res.text)
        ref = _normalize(r["ref"])
        lines = res.meta.get("page_results", [{}])[0].get("lines", [])
        best_iou = 0.0
        for ln in lines:
            box = ln.get("box")
            if box:
                best_iou = max(best_iou, _iou(r["bbox"], box))
        per_line.append({
            "ref": ref,
            "hyp": hyp,
            "exact": hyp == ref,
            "cer": round(_cer(ref, hyp), 4),
            "wer": round(_wer(ref, hyp), 4),
            "iou": round(best_iou, 4),
            "confidence": round(res.confidence, 4),
        })

    n = len(per_line)
    summary = {
        "golden_set": GOLDEN_VERSION,
        "indra_ocr": getattr(indra_ocr, "__version__", "1.1.2"),
        "lines": n,
        "exact_matches": sum(x["exact"] for x in per_line),
        "mean_cer": round(sum(x["cer"] for x in per_line) / n, 4),
        "mean_wer": round(sum(x["wer"] for x in per_line) / n, 4),
        "line_recall": round(sum(1 for x in per_line if x["hyp"]) / n, 4),
        "mean_iou": round(sum(x["iou"] for x in per_line) / n, 4),
        "mean_confidence": round(sum(x["confidence"] for x in per_line) / n, 4),
        "per_line": per_line,
    }
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    assert summary["mean_cer"] <= 0.06, summary
    assert summary["mean_wer"] <= 0.30, summary
    assert summary["line_recall"] >= 0.95, summary
    assert summary["mean_iou"] >= 0.85, summary
    assert summary["exact_matches"] >= 6, summary
