"""Regression tests: offline environment must be set before importing OCR engines."""
from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_model_store.py"
SPEC = importlib.util.spec_from_file_location("smoke_test_model_store_order_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

OFFLINE_FLAGS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


class _SyntheticImage:
    def save(self, path):
        Path(path).write_bytes(b"synthetic-image")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _ImageAPI:
    @staticmethod
    def new(*_args, **_kwargs):
        return _SyntheticImage()

    @staticmethod
    def open(*_args, **_kwargs):
        return _SyntheticImage()


@pytest.mark.parametrize("engine_name", ["onnx", "jina"])
def test_main_sets_offline_flags_before_importing_engine(
    monkeypatch, capsys, tmp_path, engine_name
):
    for key in OFFLINE_FLAGS:
        monkeypatch.delenv(key, raising=False)

    closed = []
    class StoreStub:
        def __init__(self, _root):
            pass

        def get_verified_file(self, *_args):
            return tmp_path / "model.onnx"

        def get_verified(self, *_args):
            return tmp_path / "model"

    class EngineStub:
        def __init__(self, **_kwargs):
            pass

        def _ensure_loaded(self):
            pass

        def detect(self, _image):
            return []

        def extract(self, _path):
            return SimpleNamespace(markdown="")

        def close(self):
            closed.append(True)

    engine_module = (
        "ocr_core.preprocess.word_detector"
        if engine_name == "onnx"
        else "ocr_core.engines.jina_vlm"
    )
    engine_class = "OnnxWordDetector" if engine_name == "onnx" else "JinaVLMEngine"
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PIL":
            assert all(__import__("os").environ.get(k) == v for k, v in OFFLINE_FLAGS.items())
            return SimpleNamespace(Image=_ImageAPI)
        if name == "ocr_core.model_store":
            assert all(__import__("os").environ.get(k) == v for k, v in OFFLINE_FLAGS.items())
            return SimpleNamespace(ModelStore=StoreStub)
        if name == engine_module:
            # This is the critical ordering assertion: importing the provider
            # adapter must happen only after the helper has forced offline mode.
            assert all(__import__("os").environ.get(k) == v for k, v in OFFLINE_FLAGS.items())
            return SimpleNamespace(**{engine_class: EngineStub})
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--store", str(tmp_path / "store"),
            "--model-id", "synthetic/model",
            "--revision", "a" * 40,
            "--engine", engine_name,
        ],
    )

    assert MODULE.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "loaded_and_inferred"
    assert report["network_fallback"] is False
    assert closed == [True]
