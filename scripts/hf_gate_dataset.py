#!/usr/bin/env python3
"""أداة gating مجموعة التصحيحات الطبية — OCR-04 (ocr-core#76) — للمالك فقط.

المهمة العاجلة من القضية: إيقاف النشر العام لـ
``DrAbdulmalek/arabic-medical-ocr-corrections`` (319 سجلًا من تقارير طبية)
بتحويله إلى gated فورًا — لا يبقى عامًا بينما «نخطط للمراجعة».

⚠️ تحتاج توكن HuggingFace بصلاحية كتابة على الحساب — يُقرأ من متغير البيئة
``HF_TOKEN`` فقط (لا يُقبل على سطر الأوامر، ولا يُطبع أبدًا).

الاستخدام:
    HF_TOKEN=hf_xxx python3 scripts/hf_gate_dataset.py                # gated=manual
    HF_TOKEN=hf_xxx python3 scripts/hf_gate_dataset.py --mode private # إخفاء تام
    python3 scripts/hf_gate_dataset.py --verify-only                  # فحص الحالة فقط
    python3 scripts/hf_gate_dataset.py --help

بدون توكن: يطبع خطوات الـUI اليدوية الدقيقة ويخرج بكود 2 (لا يفعل شيئًا).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.request import Request, urlopen

REPO_ID = "DrAbdulmalek/arabic-medical-ocr-corrections"


def _dataset_state(token: str | None) -> dict:
    """الحالة الحالية للـdataset عبر HF API — بلا شبكة لا شيء هنا."""
    req = Request(f"https://huggingface.co/api/datasets/{REPO_ID}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {
        "id": data.get("id"),
        "private": bool(data.get("private")),
        "gated": data.get("gated"),
        "downloads": data.get("downloads"),
    }


def _apply_mode(token: str, mode: str) -> dict:
    """تطبيق الوضع عبر huggingface_hub — استيراد مؤجَّل، رسالة خطأ واضحة."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ مكتبة huggingface_hub غير مثبتة:  pip install -U huggingface_hub")
        raise SystemExit(2)

    api = HfApi(token=token)
    before = _dataset_state(token)
    if mode == "private":
        api.update_repo_visibility(repo_id=REPO_ID, repo_type="dataset", private=True)
    else:  # gated
        api.update_repo_settings(repo_id=REPO_ID, repo_type="dataset", gated="manual")
    after = _dataset_state(token)
    return {"before": before, "after": after}


def main() -> int:
    parser = argparse.ArgumentParser(description="gating dataset التصحيحات الطبية (OCR-04)")
    parser.add_argument("--mode", choices=("gated", "private"), default="gated",
                        help="gated=طلب قبول للنزّلين (الافتراضي) | private=إخفاء تام")
    parser.add_argument("--verify-only", action="store_true", help="فحص الحالة فقط بلا تغيير")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    state = _dataset_state(token or None)
    print(f"الحالة الحالية: {json.dumps(state, ensure_ascii=False)}")

    already = (args.mode == "private" and state["private"]) or (
        args.mode == "gated" and state["gated"] not in (False, None, "False")
    )
    if args.verify_only or already:
        verdict = "مؤمَّنة بالفعل ✓" if already else "للمراقبة فقط"
        print(f"{verdict}: {REPO_ID}")
        return 0

    if state["private"] is False and state["gated"] in (False, None, "False"):
        print("⚠️  الـdataset ما زال عامًا وغير gated — القضية #76 تصنّفه عاجلًا.")

    if not token:
        print(
            "\nلا يوجد HF_TOKEN في البيئة — نفّذ أحد أمرين:\n"
            f"  (أ) آليًا:  HF_TOKEN=hf_xxx python3 scripts/hf_gate_dataset.py --mode {args.mode}\n"
            f"  (ب) يدويًا: huggingface.co/datasets/{REPO_ID} → Settings →\n"
            "      «Who can view?» → Private أو «Gating» → Enable (Request access = manual)\n"
            "ثم تحقق: python3 scripts/hf_gate_dataset.py --verify-only"
        )
        return 2

    result = _apply_mode(token, args.mode)
    after = result["after"]
    ok = after["private"] or after["gated"] not in (False, None, "False")
    print(f"قبل: {json.dumps(result['before'], ensure_ascii=False)}")
    print(f"بعد: {json.dumps(after, ensure_ascii=False)}")
    if ok:
        print(f"✅ تم {args.mode} — {REPO_ID} لم يعد عامًا.")
        print("التذكير (معايير القبول في #76): المراجعة اليدوية للسجلات على المالك"
              " قبل أي إعادة نشر — scripts/privacy_selftest.py")
        return 0
    print("❌ الحالة بعد التنفيذ لم تتحقق — راجع توكن HF وصلاحياته.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
