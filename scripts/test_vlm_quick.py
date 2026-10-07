#!/usr/bin/env python3
"""اختبار حيوي سريع لـ jina-ocr-v1.

الاستخدام:
    pip install "marathon-ocr-core[jina]"
    python scripts/test_vlm_quick.py --image doc.png
    python scripts/test_vlm_quick.py --arabic   # اختبار العربية
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--arabic", action="store_true",
                        help="اختبر بصورة عربية مُصنَّعة")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    from ocr_core.engines.jina_vlm import (
        JinaVLMEngine, STRICT_OCR_PROMPT, is_available,
    )

    if not is_available():
        print("❌ التبعيات مفقودة. ثبّت:")
        print("   pip install 'marathon-ocr-core[jina]'")
        return 1

    # تحضير الصورة
    if args.arabic:
        from PIL import Image, ImageDraw
        tmp = Path("/tmp/_arabic_test.png")
        img = Image.new("RGB", (800, 300), "white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), "السلام عليكم ورحمة الله وبركاته", fill="black")
        img.save(tmp)
        image_path = str(tmp)
        print(f"→ صورة عربية مُصنَّعة: {tmp}")
    elif args.image:
        image_path = args.image
    else:
        print("❌ حدد --image أو --arabic")
        return 1

    # استخراج
    print("→ تحميل jina-ocr-v1 (قد يأخذ دقائق أول مرة)...")
    engine = JinaVLMEngine()

    print(f"→ استخراج من {image_path}...")
    start = time.perf_counter()
    prompt = STRICT_OCR_PROMPT if args.strict else None
    result = engine.extract(image_path, prompt=prompt)
    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 60}")
    print(f"✅ اكتمل في {elapsed:.2f}s")
    print(f"{'=' * 60}\n")
    print(result.markdown)

    # تحليل
    import re
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", result.markdown))
    print(f"\n{'=' * 60}")
    print(f"📊 إحصاءات:")
    print(f"   طول Markdown: {len(result.markdown)} حرف")
    print(f"   حروف عربية: {arabic_chars}")
    print(f"   زمن التنفيذ: {elapsed:.2f}s")
    print(f"{'=' * 60}")

    # احفظ
    out = Path(image_path).with_suffix(".md")
    out.write_text(result.markdown, encoding="utf-8")
    print(f"💾 {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
