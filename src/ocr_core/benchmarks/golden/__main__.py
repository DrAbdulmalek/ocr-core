"""CLI: python -m ocr_core.benchmarks.golden --out REPORT_DIR [options]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocr_core.benchmarks.golden",
        description=(
            "Run the unified golden sample (docs/09 §6): same pairs, same "
            "normalization, comparative CER table across engines and "
            "binarization variants."
        ),
    )
    parser.add_argument("--out", required=True, help="output directory for report.md/results.json/rows.csv")
    parser.add_argument(
        "--engines",
        default="tesseract,surya",
        help="comma-separated engine spec (default: tesseract,surya; unavailable ones are skipped with a note)",
    )
    parser.add_argument(
        "--tessdata-dir",
        default=None,
        help="optional tessdata directory (e.g. documented tessdata_fast models)",
    )
    parser.add_argument("--limit", type=int, default=None, help="use only the first N pairs (smoke runs)")
    parser.add_argument("--lang", default=None, choices=["ar", "en"], help="restrict to one language")
    parser.add_argument("--condition", default=None, choices=["clean", "shadow", "low_contrast", "noisy"], help="restrict to one condition")
    args = parser.parse_args(argv)

    from ocr_core.postprocess.normalization import load_medical_dict

    from .dataset import golden_pairs
    from .report import aggregate, save_outputs
    from .runner import VARIANTS, build_engines, run_golden_benchmark

    load_medical_dict()  # prime the bundled medical dictionary once

    engines, notes = build_engines(
        args.engines.split(","), tessdata_dir=args.tessdata_dir
    )
    print("== engine availability ==")
    for n in notes:
        print(f"  {n}")
    if not engines:
        print("no engine available — aborting", file=sys.stderr)
        return 2

    pairs = golden_pairs()
    if args.lang:
        pairs = [p for p in pairs if p.lang == args.lang]
    if args.condition:
        pairs = [p for p in pairs if p.condition == args.condition]
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"== running {len(pairs)} pairs × {len(VARIANTS)} variants × {len(engines)} engine(s) ==")
    result = run_golden_benchmark(engines=engines, pairs=pairs)

    agg = aggregate(result["rows"])
    paths = save_outputs(result["rows"], agg, result["meta"], Path(args.out))

    print("\n== report ==")
    from .report import to_markdown

    print(to_markdown(agg, result["meta"]))
    print("saved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
