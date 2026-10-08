#!/usr/bin/env python3
"""golden_harness.py — أداة القياس الذهبي الموثوقة (F-13/F-16).

العقد الحاكم ضد التطبيع المزدوج:
    1. طبّع الطرفين (المرجع والفرضية) عبر ``ocr_core.eval.normalize_v1``
       — نقطة تطبيع واحدة مُعلنة الإصدار.
    2. استدعِ ``calculate_cer`` / ``calculate_wer`` مع
       ``skip_internal_normalize=True``.

أي رقم CER/WER منشور يجب أن يأتي من هذا المسار (أو مسار مطابق موثق) —
أي رقم قبله غير موثوق (F-13/F-16).

الاستخدام:
    python scripts/golden_harness.py pairs.jsonl [--json]

صيغة المدخلات (JSONL — سطر لكل زوج):
    {"id": "اختياري", "reference": "النص المرجعي", "hypothesis": "نص OCR"}

المخرجات: جدول لكل زوج (id, CER, WER) + تجميع (متوسط، أقصى)؛
و--json لمخرجات آلة قابلة للتسلسل. لا تبعيات خارجية (stdlib فقط).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# إتاحة الحزمة عند التشغيل من شجرة المصدر مباشرة
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_core.eval.metrics import calculate_cer, calculate_wer
from ocr_core.eval.normalize import NORMALIZE_V1_VERSION, normalize_v1


def load_pairs(path: Path) -> list[dict]:
    """يحمّل أزواج (reference, hypothesis) من ملف JSONL مع تحقق صارم."""
    pairs = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"سطر {lineno}: JSON غير صالح: {exc}")
        if not isinstance(rec, dict) or "reference" not in rec or "hypothesis" not in rec:
            raise SystemExit(f"سطر {lineno}: كل سجل يحتاج 'reference' و'hypothesis'")
        pairs.append(rec)
    if not pairs:
        raise SystemExit("لا توجد أزواج في المدخل")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="قياس CER/WER بعد تطبيع normalize_v1 للطرفين")
    parser.add_argument("pairs", type=Path, help="ملف JSONL: {reference, hypothesis[, id]}")
    parser.add_argument("--json", action="store_true", help="مخرجات JSON بدل الجدول")
    parser.add_argument("--fold-hamza", action="store_true",
                        help="تفعيل طي الهمزات الاختياري (يُسجّل في provenance)")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)

    rows = []
    cers, wers = [], []
    for idx, rec in enumerate(pairs, 1):
        ref_norm, ref_policy = normalize_v1(rec["reference"], fold_hamza=args.fold_hamza)
        hyp_norm, hyp_policy = normalize_v1(rec["hypothesis"], fold_hamza=args.fold_hamza)
        cer, cer_edits, cer_total = calculate_cer(ref_norm, hyp_norm, skip_internal_normalize=True)
        wer, wer_edits, wer_total = calculate_wer(ref_norm, hyp_norm, skip_internal_normalize=True)
        cers.append(cer)
        wers.append(wer)
        rows.append({
            "id": rec.get("id") or f"pair-{idx:03d}",
            "cer": round(cer, 6),
            "wer": round(wer, 6),
            "cer_edits": cer_edits,
            "cer_total": cer_total,
            "wer_edits": wer_edits,
            "wer_total": wer_total,
            "normalize": {
                "reference": {"version": ref_policy["version"], "fold_hamza": ref_policy["fold_hamza"]},
                "hypothesis": {"version": hyp_policy["version"], "fold_hamza": hyp_policy["fold_hamza"]},
            },
        })

    aggregate = {
        "pairs": len(rows),
        "normalize_version": NORMALIZE_V1_VERSION,
        "skip_internal_normalize": True,
        "fold_hamza": bool(args.fold_hamza),
        "cer_mean": round(statistics.fmean(cers), 6) if cers else None,
        "cer_max": round(max(cers), 6) if cers else None,
        "wer_mean": round(statistics.fmean(wers), 6) if cers else None,
        "wer_max": round(max(wers), 6) if cers else None,
    }

    if args.json:
        print(json.dumps({"aggregate": aggregate, "items": rows},
                         ensure_ascii=False, indent=2))
    else:
        print(f"normalize={NORMALIZE_V1_VERSION}  skip_internal_normalize=True  "
              f"pairs={aggregate['pairs']}")
        print(f"{'id':<16} {'CER':>8} {'WER':>8}")
        for r in rows:
            print(f"{r['id']:<16} {r['cer']:>8.4f} {r['wer']:>8.4f}")
        print(f"{'MEAN':<16} {aggregate['cer_mean']:>8.4f} {aggregate['wer_mean']:>8.4f}")
        print(f"{'MAX':<16} {aggregate['cer_max']:>8.4f} {aggregate['wer_max']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
