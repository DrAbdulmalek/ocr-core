#!/usr/bin/env python3
"""check_surya_api.py — فحص حي لواجهة Surya المثبتة في بيئتك.

القاعدة الذهبية: لا تفترض APIs المكتبات — تحقق منها.
المخرجات المتوقعة أدناه موثقة من فحص حي لـ surya-ocr 0.22.1 (2026-10-05).
إن اختلفت المخرجات عن "المتوقع"، عدّل surya_engine.py ليطابق بيئتك — لا العكس.

الاستخدام:
    pip install surya-ocr   # أو pip install -e ".[surya]"
    python scripts/check_surya_api.py
"""
from __future__ import annotations

import importlib
import inspect
import sys

EXPECTED = {
    "surya.detection": ["DetectionPredictor"],
    "surya.recognition": ["RecognitionPredictor"],
    "surya.layout": ["LayoutPredictor"],
    "surya.table_rec": ["TableRecPredictor"],
}

EXPECTED_CALLS = {
    "surya.detection.DetectionPredictor.__call__": "(self, images, batch_size=None, include_maps=False)",
    "surya.recognition.RecognitionPredictor.__call__": "(self, images, layout_results=None, *, full_page=None)",
    "surya.layout.LayoutPredictor.__call__": "(self, images, target_image_sizes=None, max_tokens=None)",
}

EXPECTED_SCHEMAS = {
    "TextDetectionResult": ["bboxes", "image_bbox"],
    "PageOCRResult": ["blocks", "image_bbox"],
    "BlockOCRResult": ["label", "reading_order", "html", "polygon", "confidence"],
    "LayoutResult": ["bboxes", "image_bbox"],
}


def main() -> int:
    try:
        import surya
    except ImportError:
        print("❌ surya غير مثبت. ثبّته: pip install -e \".[surya]\"")
        return 1

    version = getattr(surya, "__version__", "unknown")
    print(f"✅ surya مثبت — version: {version}")
    if version.startswith("0.22"):
        print("   (هذا المحرك مستهدف 0.22.x — ممتاز)")
    else:
        print(f"   ⚠️ الإصدار {version} قد يختلف عن 0.22.x المُتحقق منه")

    failures = 0

    for mod_name, classes in EXPECTED.items():
        try:
            mod = importlib.import_module(mod_name)
            found = [n for n, _ in inspect.getmembers(mod, inspect.isclass)]
            missing = [c for c in classes if c not in found]
            if missing:
                print(f"  ❌ {mod_name}: مفقود {missing}")
                failures += 1
            else:
                print(f"  ✅ {mod_name}: {classes}")
        except ImportError as exc:
            print(f"  ❌ {mod_name} — {exc}")
            failures += 1

    print("\n== توقيعات __call__ ==")
    for dotted, expected_sig in EXPECTED_CALLS.items():
        mod_name, _, cls_name, _, method = dotted.rpartition(".")
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
            sig = str(inspect.signature(getattr(cls, method)))
            # تطبيع المسافات للمقارنة
            norm = lambda s: " ".join(s.split())  # noqa: E731
            if norm(expected_sig) in norm(sig):
                print(f"  ✅ {dotted}{sig}")
            else:
                print(f"  ⚠️ {dotted}{sig}")
                print(f"      المتوقع (0.22.1): {expected_sig}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {dotted} — {exc}")
            failures += 1

    print("\n== مخططات النتائج ==")
    schema_pairs = [
        ("surya.detection.schema", "TextDetectionResult", EXPECTED_SCHEMAS["TextDetectionResult"]),
        ("surya.recognition.schema", "PageOCRResult", EXPECTED_SCHEMAS["PageOCRResult"]),
        ("surya.recognition.schema", "BlockOCRResult", EXPECTED_SCHEMAS["BlockOCRResult"]),
        ("surya.layout.schema", "LayoutResult", EXPECTED_SCHEMAS["LayoutResult"]),
    ]
    for mod_name, cls_name, fields in schema_pairs:
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
            names = set(getattr(cls, "model_fields", {}).keys())
            missing = [f for f in fields if f not in names]
            if missing:
                print(f"  ⚠️ {cls_name}: حقول مفقودة {missing} (متاح: {sorted(names)})")
            else:
                print(f"  ✅ {cls_name}: كل الحقول المتوقعة موجودة")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {cls_name} — {exc}")
            failures += 1

    print()
    if failures:
        print(f"النتيجة: {failures} اختلافًا — عدّل surya_engine.py ليطابق بيئتك.")
        return 1
    print("النتيجة: الواجهة مطابقة للمُتحقق منه 0.22.1 — المحرك جاهز.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
