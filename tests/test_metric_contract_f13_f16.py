# tests/test_metric_contract_f13_f16.py
"""حرس عقد القياس F-13/F-16 — التطبيع مرة واحدة، ثم قياس بلا تطبيع داخلي.

العقد المُوثّق (S2-T3 / F-13):

    ref_n, _ = normalize_v1(reference)
    hyp_n, _ = normalize_v1(hypothesis)
    cer, _, _ = calculate_cer(ref_n, hyp_n, skip_internal_normalize=True)

المسار المُوثّق هو المرجع لكل رقم منشور في المنظومة، و`omni-medical-suite`
يطبّقه عبر `packages/evaluation/golden_harness.py` بعد #185. هذا الملف يُثبت
تنفيذيًا أن المسار قابل للقياس وأن التطبيع المزدوج يُنتج الرقم نفسه (وإلا
فالأرقام غير قابلة للمقارنة بين المستودعات).

اختبار حتمي، خارجي، بلا شبكة وبلا نماذج.
"""
from __future__ import annotations

import pytest

ev = pytest.importorskip("ocr_core.eval", reason="النواة القانونية للقياس غير متاحة في هذا الإصدار")

REFERENCE = "التشخيص الطبي: ١٢٣ ملم — تم التأكيد ✓"
HYPOTHESIS = "التشخيص الطبي: 123 ملم - تم التاكيد"
TOL = 1e-9


def _scalar(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _norm(text: str) -> str:
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
        pytest.skip(f"توقيع {name} لا يقبل {sorted(kw)}")


def test_normalize_v1_is_idempotent():
    n = _norm(REFERENCE)
    assert isinstance(n, str) and n
    assert _norm(n) == n, "تطبيع المطبَّع يجب ألا يغيّر النص"


def test_identical_pair_scores_zero():
    n = _norm(REFERENCE)
    assert _metric("calculate_cer", n, n, skip_internal_normalize=True) == 0
    assert _metric("calculate_wer", n, n, skip_internal_normalize=True) == 0


CONTRACT_MODE = "AB"


def test_documented_contract_path_matches_auto_normalizing_path():
    """المسار المُوثّق (تطبيع مسبق + skip_internal_normalize) يجب أن يعطي رقم
    مسار التطبيع التلقائي على النص الخام — أي فرق يعني تطبيعًا مزدوجًا أو
    تطبيعًا مختلفًا، وأرقامًا غير قابلة للمقارنة بين المستودعات."""
    if CONTRACT_MODE != "AB":
        pytest.skip("هذا الإصدار يطبّق المقارنة B==C بدلًا من A==B")
    for name in ("calculate_cer", "calculate_wer"):
        raw = _metric(name, REFERENCE, HYPOTHESIS)
        documented = _metric(name, _norm(REFERENCE), _norm(HYPOTHESIS), skip_internal_normalize=True)
        assert abs(raw - documented) < TOL, (
            f"{name}: المسار المُوثّق ({documented!r}) لا يساوي مسار التطبيع التلقائي "
            f"({raw!r}) — خطر تطبيع مزدوج وأرقام غير قابلة للمقارنة"
        )


def test_already_normalized_input_is_not_normalized_twice():
    if CONTRACT_MODE != "CB":
        pytest.skip("هذا الإصدار يحقّق A==B مباشرة")
    for name in ("calculate_cer", "calculate_wer"):
        once = _metric(name, _norm(REFERENCE), _norm(HYPOTHESIS))
        doc = _metric(name, _norm(REFERENCE), _norm(HYPOTHESIS), skip_internal_normalize=True)
        assert abs(once - doc) < TOL, f"{name}: تطبيع مزدوج يغيّر الرقم"


def test_published_numbers_are_bounded():
    for name in ("calculate_cer", "calculate_wer"):
        v = _metric(name, _norm(REFERENCE), _norm(HYPOTHESIS), skip_internal_normalize=True)
        assert 0.0 <= v <= 1.0, f"{name} خارج المجال [0,1]: {v}"
