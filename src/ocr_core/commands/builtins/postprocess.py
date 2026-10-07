"""أوامر ما بعد المعالجة — تغلّف ArabicMedicalCorrections والميثاق."""
from ..core import command, ExecutionContext


@command(
    id="postprocess.correct_ar", name="تصحيح عربي طبي",
    description="يطبّق التصحيحات الطبية العربية على النص.",
    category="postprocess",
    params_schema={"type": "object", "properties": {
        "input_key": {"type": "string"}, "output_key": {"type": "string"}}, "required": []},
)
def correct_ar(ctx: ExecutionContext, input_key: str = "ocr_result",
                output_key: str = "corrected_text") -> dict:
    from ...postprocess.corrections_ar import ArabicMedicalCorrections
    val = ctx.get(input_key)
    text = val.get("text", "") if isinstance(val, dict) else str(val or "")
    corrector = ArabicMedicalCorrections()
    corrected, count = corrector.apply(text)
    ctx.set(output_key, corrected)
    return {"corrections_applied": count, "length": len(corrected)}


@command(
    id="postprocess.visual_evidence", name="تحليل الدليل البصري",
    description="يطبّق ميثاق v2 (رموز/ألوان/عدم يقين).",
    category="postprocess",
    params_schema={"type": "object", "properties": {
        "input_key": {"type": "string"}, "output_key": {"type": "string"}}, "required": []},
)
def visual_evidence(ctx: ExecutionContext, input_key: str = "ocr_result",
                     output_key: str = "visual_analysis") -> dict:
    from ...rules.engine import OCRProcessor
    val = ctx.get(input_key)
    markers = val.get("markers", []) if isinstance(val, dict) else []
    text = val.get("text", "") if isinstance(val, dict) else str(val or "")
    processor = OCRProcessor()
    analysis = processor.apply_visual_rules(text, {"detected": markers})
    ctx.set(output_key, analysis)
    return analysis


@command(
    id="postprocess.normalize_ar", name="تطبيع نص عربي",
    description="تطبيع غير مدمّر (للمطابقة فقط).",
    category="postprocess",
    params_schema={"type": "object", "properties": {
        "input_key": {"type": "string"}, "output_key": {"type": "string"}}, "required": []},
)
def normalize_ar(ctx: ExecutionContext, input_key: str = "corrected_text",
                  output_key: str = "normalized_text") -> dict:
    import re
    val = ctx.get(input_key, "")
    text = val if isinstance(val, str) else str(val or "")
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    ctx.set(output_key, text)
    return {"normalized_length": len(text)}


@command(
    id="postprocess.guardrails", name="حواجز الأرقام",
    description="يتحقق أن الأرقام الطبية لم تتغير بعد التصحيح.",
    category="postprocess",
    params_schema={"type": "object", "properties": {
        "original_key": {"type": "string"}, "corrected_key": {"type": "string"},
        "output_key": {"type": "string"}}, "required": []},
)
def guardrails(ctx: ExecutionContext, original_key: str = "ocr_result",
                corrected_key: str = "corrected_text",
                output_key: str = "guardrail_result") -> dict:
    import re
    orig = ctx.get(original_key)
    orig_text = orig.get("text", "") if isinstance(orig, dict) else str(orig or "")
    corr = ctx.get(corrected_key, "")
    corr_text = corr if isinstance(corr, str) else str(corr or "")
    digits_re = re.compile(r"[0-9\u0660-\u0669]+")
    orig_digits = digits_re.findall(orig_text)
    corr_digits = digits_re.findall(corr_text)
    ok = orig_digits == corr_digits
    result = {"digits_preserved": ok,
              "original_digits": orig_digits[:20],
              "corrected_digits": corr_digits[:20]}
    if not ok:
        ctx.set(corrected_key, orig_text)
        result["reverted"] = True
        result["reason"] = "NUMERICAL_DRIFT_DETECTED"
    ctx.set(output_key, result)
    return result


@command(
    id="postprocess.fuzzy_match_glossary", name="مطابقة ضبابية للمسرد",
    description="يقترح مصطلحات طبية قريبة من النص.",
    category="postprocess",
    params_schema={"type": "object", "properties": {
        "text_key": {"type": "string"}, "max_suggestions": {"type": "integer"},
        "output_key": {"type": "string"}}, "required": []},
)
def fuzzy_match_glossary(ctx: ExecutionContext, text_key: str = "corrected_text",
                          max_suggestions: int = 5,
                          output_key: str = "glossary_suggestions") -> dict:
    def lev(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    val = ctx.get(text_key, "")
    text = val if isinstance(val, str) else str(val or "")
    words = set(text.split())
    try:
        from ...postprocess.corrections_ar import ArabicMedicalCorrections
        glossary = set(ArabicMedicalCorrections().corrections.keys())
    except Exception:
        glossary = set()
    suggestions = []
    for word in words:
        if len(word) < 3:
            continue
        best, best_dist = None, 3
        for term in glossary:
            if abs(len(term) - len(word)) > 2:
                continue
            d = lev(word, term)
            if d < best_dist:
                best_dist, best = d, term
        if best and best != word:
            suggestions.append({"word": word, "suggested": best, "distance": best_dist})
        if len(suggestions) >= max_suggestions:
            break
    ctx.set(output_key, suggestions)
    return {"suggestions_count": len(suggestions)}
