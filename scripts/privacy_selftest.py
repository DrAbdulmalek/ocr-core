#!/usr/bin/env python3
"""اختبار ذاتي للخصوصية على سجلات حقيقية — للمالك فقط، محليًا حصرًا (OCR-04/#76).

معيار القبول في القضية: «ocr_core.privacy يجتاز اختبار الـ10 سجلات» — السجلات
الحقيقية لا تدخل git ولا CI ولا أي صندوق وكيل؛ هذا السكربت يشغّلها المالك
على جهازه على ملف jsonl محلي (مثل تصدير corrections.jsonl من الـdataset)،
ويعيد: عدّادات الحجب + فحص البقايا لكل سجل + حكمًا نهائيًا — **بلا طباعة
أي محتوى خام من السجلات إطلاقًا**.

الاستخدام:
    python3 scripts/privacy_selftest.py --records /path/corrections.jsonl
    python3 scripts/privacy_selftest.py --records x.jsonl --limit 10 --ner
    python3 scripts/privacy_selftest.py --records x.jsonl --out report.json

الخروج: 0 = لا بقايا Regex مع NER محلي محمّل (ولا يزال التدقيق البشري إلزاميًا)
| 1 = بقايا Regex | 2 = فحص غير مكتمل/خطأ إدخال.
غياب NER أو عدم طلبه لا ينتج حكم «نظيف»؛ التقرير لا يثبت خلو البيانات من PII.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_core.privacy import HybridPIIScrubber


def _load_records(path: Path, limit: int) -> list:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def _verdict(residue_total: int, *, ner_requested: bool, ner_loaded: bool) -> tuple[str, int]:
    """Never label a regex-only or NER-unavailable scan as clean."""
    if residue_total:
        return "regex_residue_detected", 1
    if ner_requested and ner_loaded:
        return "no_detected_regex_residue_ner_active_manual_review_required", 0
    return "regex_only_incomplete_manual_review_required", 2


def main() -> int:
    parser = argparse.ArgumentParser(description="اختبار الـ10 سجلات الحقيقية (محليًا فقط)")
    parser.add_argument("--records", required=True, help="مسار ملف jsonl محلي للسجلات")
    parser.add_argument("--limit", type=int, default=10, help="عدد السجلات (افتراضي 10)")
    parser.add_argument("--text-key", default="text", help="مفتاح حقل النص في السجل")
    parser.add_argument("--ner", action="store_true",
                        help="تفعيل طبقة NER إن توفّر النموذج محليًا (أبطأ)")
    parser.add_argument("--out", help="مسار تقرير JSON (عدّادات فقط)")
    args = parser.parse_args()

    path = Path(args.records).expanduser()
    if not path.is_file():
        print(f"❌ الملف غير موجود: {path}")
        return 2

    if args.limit < 0:
        print("❌ --limit يجب أن يكون صفرًا (بلا حد) أو عددًا موجبًا")
        return 2
    try:
        records = _load_records(path, args.limit)
    except (OSError, UnicodeError, json.JSONDecodeError):
        print("❌ تعذر قراءة JSONL؛ راجع سلامة الملف وترميزه دون مشاركة محتواه")
        return 2
    if not records:
        print("❌ لا سجلات في الملف")
        return 2
    if any(not isinstance(record, dict) for record in records):
        print("❌ يجب أن يكون كل سطر كائن JSON")
        return 2
    missing_text = sum(not isinstance(record.get(args.text_key), str) for record in records)
    if missing_text:
        print(f"❌ الحقل النصي المطلوب غير موجود أو غير نصي في {missing_text} سجل/سجلات")
        return 2

    scrubber = HybridPIIScrubber(use_ner=args.ner)
    scrubbed, stats = scrubber.scrub_records(records, text_keys=(args.text_key,), check=True)

    print(f"السجلات: {len(records)} | نصها مجموعه "
          f"{sum(len(r.get(args.text_key) or '') for r in records)} حرفًا (لا يُطبع المحتوى)")
    print(f"عدّادات الحجب المجمعة: {json.dumps(stats.to_dict(), ensure_ascii=False)}")

    residue_total = sum(stats.residue.values())
    verdict, exit_code = _verdict(
        residue_total,
        ner_requested=args.ner,
        ner_loaded=scrubber._ner_loaded,
    )
    print(f"الحكم: {verdict} (manual_review_required=true)")
    if not args.ner:
        print("⚠️ هذا فحص Regex فقط وليس تحققًا كاملًا من الأسماء أو بقية PII.")
    elif not scrubber._ner_loaded:
        print("⚠️ لم يُحمّل NER محليًا؛ النتيجة غير مكتملة. لا يحدث تنزيل شبكي.")

    if args.out:
        report = {
            "records": len(records),
            "text_key": args.text_key,
            "ner_requested": args.ner,
            "ner_loaded": scrubber._ner_loaded,
            "manual_review_required": True,
            "stats": stats.to_dict(),
            "verdict": verdict,
        }
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"التقرير (عدّادات فقط، بلا محتوى): {args.out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
