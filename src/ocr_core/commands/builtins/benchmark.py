from ..core import command, ExecutionContext


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@command(
    id="benchmark.cer", name="حساب CER",
    description="معدل خطأ الأحرف مقابل نص مرجعي.",
    category="benchmark",
    params_schema={"type": "object", "properties": {
        "prediction_key": {"type": "string"}, "reference": {"type": "string"},
        "output_key": {"type": "string"}}, "required": ["reference"]},
)
def cer(ctx: ExecutionContext, prediction_key: str = "corrected_text",
         reference: str = "", output_key: str = "cer_score") -> dict:
    pred = ctx.get(prediction_key, "")
    if not isinstance(pred, str):
        pred = pred.get("text", "") if isinstance(pred, dict) else str(pred or "")
    if not reference:
        raise ValueError("النص المرجعي مطلوب")
    distance = _lev(pred, reference)
    score = distance / max(len(reference), 1)
    result = {"cer": round(score, 4), "distance": distance,
              "reference_length": len(reference)}
    ctx.set(output_key, result)
    return result


@command(
    id="benchmark.wer", name="حساب WER",
    description="معدل خطأ الكلمات مقابل نص مرجعي.",
    category="benchmark",
    params_schema={"type": "object", "properties": {
        "prediction_key": {"type": "string"}, "reference": {"type": "string"},
        "output_key": {"type": "string"}}, "required": ["reference"]},
)
def wer(ctx: ExecutionContext, prediction_key: str = "corrected_text",
         reference: str = "", output_key: str = "wer_score") -> dict:
    pred = ctx.get(prediction_key, "")
    if not isinstance(pred, str):
        pred = pred.get("text", "") if isinstance(pred, dict) else str(pred or "")
    distance = _lev(pred.split(), reference.split())
    score = distance / max(len(reference.split()), 1)
    result = {"wer": round(score, 4), "distance": distance,
              "reference_words": len(reference.split())}
    ctx.set(output_key, result)
    return result
