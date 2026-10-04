#!/usr/bin/env python3
"""fetch_tessdata.py — تنزيل قابل لإعادة الإنتاج لنماذج Tesseract مع تحقق SHA-256.

الافتراضي: tessdata_fast (المضمّن في الحزمة أصلاً — هذا السكربت للتحقق/إعادة الجلب).
بخيار --best: جلب نسخ tessdata_best عالية الدقة إلى مجلد محلي خارج الحزمة.

الاستخدام:
    python scripts/fetch_tessdata.py            # تحقق من المضمن + إعادة جلب عند التلف
    python scripts/fetch_tessdata.py --best     # جلب best إلى ./.tessdata_best (غير مُتعقب)
    python scripts/fetch_tessdata.py --langs ara eng
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
BUNDLED = SRC_ROOT / "src" / "ocr_core" / "resources" / "tessdata_fast"

# بصمات tessdata_fast @ 87416418657359cb625c412a48b6e1d6d41c29bd (2024-08-01)
FAST_SHA = {
    "ara": "e3206d3dc87fd50c24a0fb9f01838615911d25168f4e64415244b67d2bb3e729",
    "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
}
# best: تُتحقق وقت التنزيل بقياس الحد الأدنى للحجم فقط (تتغير مع التحديثات)
BEST_MIN_BYTES = {"ara": 2_000_000, "eng": 10_000_000}

BASE_FAST = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{lang}.traineddata"
BASE_BEST = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/{lang}.traineddata"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(lang: str, dest: Path, base: str, expected_sha: str | None, min_bytes: int) -> int:
    url = base.format(lang=lang)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"↓ {lang}: {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    if len(data) < min_bytes:
        print(f"  ❌ الحجم صغير جدًا ({len(data)} < {min_bytes}) — توقف")
        return 1
    dest.write_bytes(data)
    got = sha256_of(dest)
    if expected_sha:
        ok = got == expected_sha
        print(f"  sha256: {got[:16]}... {'✅ مطابق' if ok else '❌ غير مطابق للموثق'}")
        if not ok:
            return 1
    else:
        print(f"  sha256: {got[:16]}... (best — بلا بصمة موثقة، الحجم سليم)")
    print(f"  ✅ حفظ: {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", action="store_true", help="جلب tessdata_best بدل fast")
    ap.add_argument("--langs", nargs="+", default=["ara", "eng"])
    args = ap.parse_args()

    rc = 0
    if args.best:
        dest_dir = SRC_ROOT / ".tessdata_best"
        print("الوضع: best →", dest_dir, "(أضف المجلد إلى .gitignore — خارج الحزمة)")
        for lang in args.langs:
            rc |= fetch(lang, dest_dir / f"{lang}.traineddata", BASE_BEST, None, BEST_MIN_BYTES.get(lang, 1_000_000))
    else:
        print("الوضع: fast — تحقق/إصلاح المضمن في", BUNDLED)
        for lang in args.langs:
            target = BUNDLED / f"{lang}.traineddata"
            expected = FAST_SHA.get(lang)
            if target.exists() and expected and sha256_of(target) == expected:
                print(f"  ✅ {lang}: مضمن وسليم")
                continue
            rc |= fetch(lang, target, BASE_FAST, expected, 500_000)
    return rc


if __name__ == "__main__":
    sys.exit(main())
