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

الخروج: 0 = كل السجلات نظيفة بعد الحجب | 1 = بقايا تحتاج انتباهًا | 2 = خطأ إدخال.
التقرير المكتوب يحمل العدّادات فقط (لا محتوى) — يُرفق في تعليق على #76.
"""
from __future__ import annotations

import argparse
import json
import sys
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

    records = _load_records(path, args.limit)
    if not records:
        print("❌ لا سجلات في الملف")
        return 2
    if args.text_key not in records[0]:
        print(f"❌ المفتاح '{args.text_key}' غير موجود في السجل الأول — "
              f"المفاتيح المتاحة: {sorted(k for k, v in records[0].items() if isinstance(v, str))}")
        return 2

    scrubber = HybridPIIScrubber(use_ner=args.ner)
    scrubbed, stats = scrubber.scrub_records(records, text_keys=(args.text_key,), check=True)

    print(f"السجلات: {len(records)} | نصها مجموعه "
          f"{sum(len(r.get(args.text_key) or '') for r in records)} حرفًا (لا يُطبع المحتوى)")
    print(f"عدّادات الحجب المجمعة: {json.dumps(stats.to_dict(), ensure_ascii=False)}")

    residue_total = sum(stats.residue.values())
    verdict = "نظيفة ✓" if residue_total == 0 else f"بقايا: {residue_total} مطابقة — تحتاج انتباهًا"
    print(f"الحكم بعد الحجب: {verdict}")

    if args.out:
        report = {
            "records": len(records),
            "text_key": args.text_key,
            "ner": args.ner,
            "stats": stats.to_dict(),
            "verdict": "clean" if residue_total == 0 else "residue",
        }
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"التقرير (عدّادات فقط، بلا محتوى): {args.out}")

    return 0 if residue_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
