# tests/test_metric_contract_f13_f16.py
"""حرس عقد القياس F-13/F-16 — تطبيع الطرفين مرة واحدة، ثم قياس بلا تطبيع داخلي.

العقد المُوثّق (S2-T3 / F-13 / F-16):

    ref_n, _policy = normalize_v1(reference)
    hyp_n, _policy = normalize_v1(hypothesis)
    cer, _, _ = calculate_cer(ref_n, hyp_n, skip_internal_normalize=True)
    wer, _, _ = calculate_wer(ref_n, hyp_n, skip_internal_normalize=True)

أي رقم منشور في المشروع يجب أن يأتي من هذا المسار. هذا الملف لا يعيد تنفيذ
المنطق؛ بل يُثبت أن العقد قابل للتنفيذ فعليًا وأن التطبيع المزدوج لا يغيّر
الرقم المنشور (وإلا فالأرقام غير قابلة للمقارنة بين المستودعات).

اختبار حتمي، خارجي (offline)، بلا شبكة وبلا نماذج — يشتغل في CI وفي المحلي.
"""
from __future__ import annotations

import pytest

ev = pytest.importorskip("ocr_core.eval", reason="النواة القانونية للقياس غير متاحة في هذا الإصدار")

REFERENCE = "التشخيص الطبي: ١٢٣ ملم — تم التأكيد ✓"
HYPOTHESIS = "التشخيص الطبي: 123 ملم - تم التاكيد"


def _scalar(out):
    """calculate_* قد تُرجع قيمة أو حزمة (cer, subs, dels, ins) — نأخذ الأولى."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _normalize(text: str):
    fn = getattr(ev, "normalize_v1", None)
    if not callable(fn):
        pytest.skip("ocr_core.eval.normalize_v1 غير متاح")
    out = fn(text)
    return out[0] if isinstance(out, (tuple, list)) else out


def _metric(name: str, ref: str, hyp: str, **kw):
    fn = getattr(ev, name, None)
    if not callable(fn):
        pytest.skip(f"ocr_core.eval.{name} غير متاح")
    try:
        return _scalar(fn(ref, hyp, **kw))
    except TypeError:
        pytest.skip(f"توقيع {name} لا يقبل {sorted(kw)} في هذا الإصدار")


def test_normalize_v1_is_idempotent():
    n = _normalize(REFERENCE)
    assert isinstance(n, str) and n, "normalize_v1 يجب أن تُرجع نصًا غير فارغ"
    assert _normalize(n) == n, "تطبيع المطبيع نفسه يجب ألا يغيّر النص (وإلا فالتطبيع المزدوج يفسد الرقم)"


def test_identical_pair_scores_zero():
    n = _normalize(REFERENCE)
    assert _metric("calculate_cer", n, n, skip_internal_normalize=True) == 0
    assert _metric("calculate_wer", n, n, skip_internal_normalize=True) == 0


def test_cer_parity_with_and_without_internal_normalize():
    ref_n, hyp_n = _normalize(REFERENCE), _normalize(HYPOTHESIS)
    direct = _metric("calculate_cer", ref_n, hyp_n)
    documented = _metric("calculate_cer", ref_n, hyp_n, skip_internal_normalize=True)
    assert direct == pytest.approx(documented), (
        "المسار المُوثّق (تطبيع مرة واحدة + skip_internal_normalize) يجب أن يعطي "
        "الرقم نفسه؛ أي فرق يعني تطبيعًا مزدوجًا وأرقامًا غير قابلة للمقارنة"
    )


def test_wer_parity_with_and_without_internal_normalize():
    ref_n, hyp_n = _normalize(REFERENCE), _normalize(HYPOTHESIS)
    direct = _metric("calculate_wer", ref_n, hyp_n)
    documented = _metric("calculate_wer", ref_n, hyp_n, skip_internal_normalize=True)
    assert direct == pytest.approx(documented)


def test_published_numbers_are_bounded():
    ref_n, hyp_n = _normalize(REFERENCE), _normalize(HYPOTHESIS)
    for name in ("calculate_cer", "calculate_wer"):
        v = _metric(name, ref_n, hyp_n, skip_internal_normalize=True)
        assert 0.0 <= v <= 1.0, f"{name} خارج المجال [0,1]: {v}"
