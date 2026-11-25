#!/usr/bin/env python3
"""Local-only smoke test for engines backed by a verified ModelStore.

Uses a generated blank image only. It never reads patient documents, prints OCR
content, downloads model files, or uploads results.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test a verified local OCR model using a synthetic blank image."
    )
    parser.add_argument("--store", required=True, help="local ModelStore root")
    parser.add_argument("--model-id", required=True, help="manifest model_id")
    parser.add_argument("--revision", required=True, help="immutable manifest revision SHA")
    parser.add_argument("--engine", choices=("onnx", "jina"), default="onnx")
    parser.add_argument("--model-file", default="model.onnx", help="manifest path for ONNX only")
    parser.add_argument("--device", default="auto", help="Jina device (auto/cpu/cuda)")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="explicitly allow execution of model-provided Python code (review it first)",
    )
    parser.add_argument("--size", type=int, default=448, help="synthetic square image size")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.size < 32 or args.size > 4096:
        raise SystemExit("--size must be between 32 and 4096")

    from PIL import Image
    from ocr_core.model_store import ModelStore

    store = ModelStore(Path(args.store).expanduser())
    if args.engine == "onnx":
        store.get_verified_file(args.model_id, args.revision, args.model_file)
        from ocr_core.preprocess.word_detector import OnnxWordDetector
        engine = OnnxWordDetector(
            model_id=args.model_id,
            model_store_root=args.store,
            model_revision=args.revision,
            model_file=args.model_file,
        )
    else:
        store.get_verified(args.model_id, args.revision)
        from ocr_core.engines.jina_vlm import JinaVLMEngine
        engine = JinaVLMEngine(
            model_id=args.model_id,
            model_store_root=args.store,
            model_revision=args.revision,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )

    report = {
        "engine": args.engine,
        "model_id": args.model_id,
        "revision": args.revision,
        "local_store_verified": True,
        "network_fallback": False,
        "trust_remote_code": bool(args.trust_remote_code) if args.engine == "jina" else False,
    }
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="ocr-core-smoke-") as temp_dir:
            image_path = Path(temp_dir) / "synthetic-blank.png"
            Image.new("RGB", (args.size, args.size), "white").save(image_path)
            load_started = time.perf_counter()
            if args.engine == "onnx":
                boxes = engine.detect(Image.open(image_path))
                report["output_count"] = len(boxes)
            else:
                result = engine.extract(str(image_path))
                report["output_characters"] = len(result.markdown)
            report["first_inference_ms"] = round((time.perf_counter() - load_started) * 1000, 2)
        report["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        report["result"] = "loaded_and_inferred"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
