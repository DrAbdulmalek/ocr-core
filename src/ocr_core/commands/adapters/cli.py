"""CLI — ocr-core <action> [args]"""
import argparse
import json
import sys
from ..core import registry, Executor, Pipeline


def cmd_list(args):
    for c in registry.list(category=args.category):
        print(f"  {c.id:<35} [{c.category}] {c.name}")


def cmd_describe(args):
    cmd = registry.get(args.command_id)
    print(json.dumps(cmd.to_dict(), ensure_ascii=False, indent=2))


def cmd_run(args):
    params = json.loads(args.params) if args.params else {}
    executor = Executor()
    result = executor.execute(args.command_id, params, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    sys.exit(0 if result.get("ok") else 1)


def cmd_pipeline(args):
    import pathlib
    data = json.loads(pathlib.Path(args.file).read_text(encoding="utf-8"))
    pipeline = Pipeline.from_json(data)
    executor = Executor()
    result = executor.execute_pipeline(pipeline)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    sys.exit(0 if result.get("ok") else 1)


def main():
    parser = argparse.ArgumentParser(prog="ocr-core",
                                     description="نظام أوامر موحد لمعالجة OCR")
    sub = parser.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser("list", help="اسرد كل الأوامر")
    p_list.add_argument("--category")

    p_desc = sub.add_parser("describe", help="وصف أمر")
    p_desc.add_argument("command_id")

    p_run = sub.add_parser("run", help="نفّذ أمراً")
    p_run.add_argument("command_id")
    p_run.add_argument("--params")
    p_run.add_argument("--dry-run", action="store_true")

    p_pipe = sub.add_parser("pipeline", help="نفّذ خط أنابيب من ملف JSON")
    p_pipe.add_argument("file")

    args = parser.parse_args()

    # تسجيل كل builtins (io, preprocess, ocr, postprocess, export,
    # benchmark, detect, vlm_ocr) — الاستيراد وحده يكفي للتسجيل.
    from .. import builtins  # noqa

    handlers = {"list": cmd_list, "describe": cmd_describe,
                "run": cmd_run, "pipeline": cmd_pipeline}
    handlers[args.action](args)


if __name__ == "__main__":
    main()
