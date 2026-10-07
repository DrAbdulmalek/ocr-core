#!/usr/bin/env python3
"""اختبار حيوي لـ jina-ocr-v1.

الاستخدام:
  pip install "marathon-ocr-core[jina,pdf]"
  python scripts/test_jina_ocr.py --image sample.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--strict", action="store_true",
                        help="استخدم prompt أكاديمي")
    args = parser.parse_args()

    from ocr_core.engines.jina_vlm import (
        JinaVLMEngine, STRICT_OCR_PROMPT, is_available,
    )

    if not is_available():
        print("❌ التبعيات مفقودة. ثبّت:")
        print("   pip install 'marathon-ocr-core[jina]'")
        return 1

    print(f"→ تحميل jina-ocr-v1...")
    engine = JinaVLMEngine()

    prompt = STRICT_OCR_PROMPT if args.strict else None
    print(f"→ استخراج من {args.image}...")
    result = engine.extract(args.image, prompt=prompt)

    print(f"\n{'=' * 60}")
    print(f"✅ اكتمل في {result.duration_ms:.0f}ms")
    print(f"{'=' * 60}\n")
    print(result.markdown)

    # احفظ
    out = Path(args.image).with_suffix(".md")
    out.write_text(result.markdown, encoding="utf-8")
    print(f"\n💾 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
