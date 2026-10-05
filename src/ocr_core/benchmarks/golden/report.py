"""Aggregation + report generation for the unified golden sample.

Produces the comparative table docs/09 §6 asks for: mean CER (and WER)
per engine × binarization variant, split by language and by page
condition — plus the honest best-variant-per-condition readout.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

__all__ = ["aggregate", "to_markdown", "save_outputs"]


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def aggregate(rows: list[dict]) -> dict:
    """Pivot *rows* into mean CER/WER per engine × variant (+ lang/condition).

    All pivot keys are ``"|"``-joined strings (JSON-serialisable):
    ``overall["tesseract|otsu"]``, ``by_lang["tesseract|otsu|ar"]``,
    ``by_condition["tesseract|auto|shadow"]``.
    """
    ok_rows = [r for r in rows if r["ok"] and r["cer"] is not None]

    def _pivot(key_fields: tuple[str, ...]) -> dict:
        buckets: dict[str, dict] = defaultdict(
            lambda: {"cer": [], "wer": [], "n": 0, "errors": 0}
        )
        for r in rows:
            key = "|".join(str(r[f]) for f in key_fields)
            if r["ok"]:
                buckets[key]["cer"].append(r["cer"])
                buckets[key]["wer"].append(r["wer"])
                buckets[key]["n"] += 1
            else:
                buckets[key]["errors"] += 1
        out = {}
        for key, b in buckets.items():
            out[key] = {
                "n": b["n"],
                "errors": b["errors"],
                "mean_cer": _mean(b["cer"]),
                "mean_wer": _mean(b["wer"]),
            }
        return out

    overall = _pivot(("engine", "variant"))
    by_lang = _pivot(("engine", "variant", "lang"))
    by_condition = _pivot(("engine", "variant", "condition"))

    # best variant per condition per engine (min mean CER)
    best_variant: dict[str, dict] = {}
    per_engine_cond: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, m in by_condition.items():
        engine, _variant, condition = key.split("|", 2)
        if m["mean_cer"] is not None:
            per_engine_cond[engine][condition].append((_variant, m["mean_cer"]))
    for engine, by_cond in per_engine_cond.items():
        for condition, lst in by_cond.items():
            variant, cer = min(lst, key=lambda t: t[1])
            best_variant[f"{engine}|{condition}"] = {
                "variant": variant,
                "mean_cer": cer,
            }

    engines = sorted({r["engine"] for r in rows})
    error_counts = {
        e: sum(1 for r in rows if r["engine"] == e and not r["ok"]) for e in engines
    }

    return {
        "overall": overall,
        "by_lang": by_lang,
        "by_condition": by_condition,
        "best_variant_per_condition": dict(best_variant),
        "error_counts": error_counts,
    }


def _md_table(header: list[str], body: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in body:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def to_markdown(agg: dict, meta: dict | None = None) -> str:
    """Render the aggregated pivots as Markdown tables (docs-ready)."""
    parts: list[str] = []
    if meta:
        parts.append(
            _md_table(
                ["field", "value"],
                [
                    ["engines", ", ".join(meta["engines"])],
                    ["variants", ", ".join(meta["variants"])],
                    ["pairs", meta["n_pairs"]],
                    ["normalization", meta["normalization"]],
                ],
            )
        )
        parts.append("")

    variants = ["raw", "otsu", "sauvola", "auto"]
    engines = sorted({k.split("|")[0] for k in agg["overall"].keys()})

    parts.append("### Overall — mean CER/WER per engine × variant")
    body = []
    for e in engines:
        for v in variants:
            m = agg["overall"].get(f"{e}|{v}")
            if m:
                body.append(
                    [e, v, m["n"], m["mean_cer"], m["mean_wer"], m["errors"]]
                )
    parts.append(_md_table(["engine", "variant", "n", "mean CER", "mean WER", "errors"], body))
    parts.append("")

    parts.append("### By language")
    body = []
    for e in engines:
        for lang in ("ar", "en"):
            for v in variants:
                m = agg["by_lang"].get(f"{e}|{v}|{lang}")
                if m:
                    body.append([e, lang, v, m["n"], m["mean_cer"], m["mean_wer"]])
    parts.append(_md_table(["engine", "lang", "variant", "n", "mean CER", "mean WER"], body))
    parts.append("")

    parts.append("### By page condition")
    body = []
    conditions = sorted({k.split("|")[2] for k in agg["by_condition"].keys()})
    for e in engines:
        for c in conditions:
            for v in variants:
                m = agg["by_condition"].get(f"{e}|{v}|{c}")
                if m:
                    body.append([e, c, v, m["n"], m["mean_cer"], m["mean_wer"]])
    parts.append(_md_table(["engine", "condition", "variant", "n", "mean CER", "mean WER"], body))
    parts.append("")

    parts.append("### Best variant per condition (min mean CER)")
    body = [
        [k.split("|")[0], k.split("|")[1], v["variant"], v["mean_cer"]]
        for k, v in sorted(agg["best_variant_per_condition"].items())
    ]
    parts.append(_md_table(["engine", "condition", "best variant", "mean CER"], body))
    return "\n\n".join(parts) + "\n"


def save_outputs(rows: list[dict], agg: dict, meta: dict, out_dir: Path) -> dict:
    """Write ``rows.csv`` + ``results.json`` + ``report.md`` into *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "pair_id", "lang", "condition", "engine", "variant", "ok", "error",
        "cer", "wer", "substitutions", "deletions", "insertions",
        "ref_len", "hyp_len", "time_s",
    ]
    with open(out_dir / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    payload = {"meta": meta, "aggregate": agg, "rows": rows}
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = to_markdown(agg, meta)
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(md)

    return {"rows_csv": str(out_dir / "rows.csv"),
            "results_json": str(out_dir / "results.json"),
            "report_md": str(out_dir / "report.md")}
