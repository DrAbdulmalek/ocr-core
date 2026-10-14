# tests/test_metric_contract_f13_f16.py
"""حرس عقد القياس F-13/F-16 — التطبيع مرة واحدة، ثم قياس بلا تطبيع داخلي.

العقد المُوثّق (S2-T3 / F-13):

    ref_n, _ = normalize_v1(reference)
    hyp_n, _ = normalize_v1(hypothesis)
    cer, _, _ = calculate_cer(ref_n, hyp_n, skip_internal_normalize=True)

وضع هذا الملف بعد قياس 2026-10-14 (قرار المالك OD-12: تحويل تأكيد المساواة
إلى اختبار توثيق سلوك). المحرّك الحالي يعطي ثلاثة أرقام مختلفة لنفس الزوج
المرجعي بحسب المسار (قيم مقاسة حرفيًا على رأس main 40221b91):

    A) خام + تطبيع داخلي تلقائي        CER=0.16216216216216217  WER=0.375
    B) تطبيع مسبق + skip=True (العقد)   CER=0.10810810810810811  WER=0.375
    C) تطبيع مسبق + تطبيع داخلي (مزدوج) CER=0.08108108108108109  WER=0.25

لذا اختبار المساواة A==B أصبح xfail متسامحًا يوثّق الانحراف، بينما
اختبارات characterization تُثبّت الأرقام المقاسة الحالية بدقة — أي تغيير
مستقبلي في المحرّك سيكسر التثبيت بصوت عالٍ بدل أن يمرّ بصمت. عند توحيد
مسار المحرّك يُعاد تفعيل A==B كتوكيد صارم (احذف علامة xfail).

اختبار حتمي، خارجي، بلا شبكة وبلا نماذج.
"""
from __future__ import annotations

import pytest

ev = pytest.importorskip("ocr_core.eval", reason="النواة القانونية للقياس غير متاحة في هذا الإصدار")

REFERENCE = "التشخيص الطبي: ١٢٣ ملم — تم التأكيد ✓"
HYPOTHESIS = "التشخيص الطبي: 123 ملم - تم التاكيد"
TOL = 1e-9

# قيم المقاس الحرفي 2026-10-14 على ocr_core.eval @ main 40221b91
MEASURED_CER_A = 0.16216216216216217  # 6/37 — خام + تطبيع داخلي
MEASURED_CER_B = 0.10810810810810811  # 4/37 — المسار المُوثّق (skip=True)
MEASURED_CER_C = 0.08108108108108109  # 3/37 — تطبيع مزدوج
MEASURED_WER_AB = 0.375               # A==B في WER
MEASURED_WER_C = 0.25                 # التطبيع المزدوج يغيّر WER


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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "انحراف محرّك مُقاس 2026-10-14: CER(A)=0.16216216216216217 "
        "vs CER(B)=0.10810810810810811 على نفس الزوج المرجعي — العقد يتطلب "
        "المساواة ويُعاد تفعيل هذا التوكيد كشرط صارم بعد توحيد مسار المحرّك "
        "(قرار المالك OD-12: characterization حتىئذ)"
    ),
)
def test_documented_contract_path_matches_auto_normalizing_path():
    """[xfail موثّق] المسار المُوثّق (تطبيع مسبق + skip_internal_normalize)
    يجب أن يعطي رقم مسار التطبيع التلقائي على النص الخام — أي فرق يعني
    تطبيعًا مزدوجًا أو تطبيعًا مختلفًا، وأرقامًا غير قابلة للمقارنة بين
    المستودعات. يفشل حاليًا بانحراف مُقاس؛ انظر test_cer_path_divergence."""
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


def test_cer_path_divergence_is_characterized():
    """تثبيت السلوك المقاس: المسارات الثلاثة تعطي أرقامها بدقة 1e-9،
    والانحراف A≠B موجود. أي تغيير في المحرّك يكسر هذا التثبيت فيُعاد
    فحص العقد فورًا."""
    a = _metric("calculate_cer", REFERENCE, HYPOTHESIS)
    b = _metric(
        "calculate_cer",
        _norm(REFERENCE),
        _norm(HYPOTHESIS),
        skip_internal_normalize=True,
    )
    c = _metric("calculate_cer", _norm(REFERENCE), _norm(HYPOTHESIS))
    for label, v, expected in (
        ("A", a, MEASURED_CER_A),
        ("B", b, MEASURED_CER_B),
        ("C", c, MEASURED_CER_C),
    ):
        assert abs(v - expected) < TOL, (
            f"CER[{label}]={v!r} ≠ المقاس {expected!r} — سلوك المحرّك تغيّر؛ "
            "أعد قياس العقد وحدّث الثوابت وحالة xfail"
        )
    assert a != b, (
        "الانحراف المُقاس بين المسارين اختفى — أعد تفعيل "
        "test_documented_contract_path_matches_auto_normalizing_path كتوكيد صارم"
    )
    assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and 0.0 <= c <= 1.0


def test_wer_path_behavior_is_characterized():
    """WER: A==B (0.375) والتطبيع المزدوج C يغيّر الرقم إلى 0.25 — تثبيت المقاس."""
    a = _metric("calculate_wer", REFERENCE, HYPOTHESIS)
    b = _metric(
        "calculate_wer",
        _norm(REFERENCE),
        _norm(HYPOTHESIS),
        skip_internal_normalize=True,
    )
    c = _metric("calculate_wer", _norm(REFERENCE), _norm(HYPOTHESIS))
    assert abs(a - MEASURED_WER_AB) < TOL, f"WER[A]={a!r} ≠ المقاس {MEASURED_WER_AB!r}"
    assert abs(b - MEASURED_WER_AB) < TOL, f"WER[B]={b!r} ≠ المقاس {MEASURED_WER_AB!r}"
    assert abs(c - MEASURED_WER_C) < TOL, f"WER[C]={c!r} ≠ المقاس {MEASURED_WER_C!r}"


def test_published_numbers_are_bounded():
    for name in ("calculate_cer", "calculate_wer"):
        v = _metric(name, _norm(REFERENCE), _norm(HYPOTHESIS), skip_internal_normalize=True)
        assert 0.0 <= v <= 1.0, f"{name} خارج المجال [0,1]: {v}"
