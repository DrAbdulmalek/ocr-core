"""Offline and explicit model-resolution policy tests; no real model or network."""
from __future__ import annotations

import builtins
import socket
import sys
import types
from unittest.mock import MagicMock, Mock

import pytest
from PIL import Image, ImageDraw

from ocr_core.model_access import ModelAccessPolicy
from ocr_core.preprocess.word_detector import (
    LocalModelUnavailableError,
    OnnxWordDetector,
    ProjectionDetector,
    create_detector,
)


def _fake_jina_modules(monkeypatch, processor_side_effect=None):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.bfloat16 = object()

    processor = MagicMock()
    processor_factory = Mock(return_value=processor)
    if processor_side_effect is not None:
        processor_factory.side_effect = processor_side_effect

    model = MagicMock()
    model.to.return_value = model
    model_factory = Mock(return_value=model)

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=processor_factory)
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=model_factory)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return processor_factory, model_factory


def test_model_access_modes_fail_closed():
    assert ModelAccessPolicy().mode == "LOCAL_ONLY"
    assert ModelAccessPolicy().local_files_only is True
    assert ModelAccessPolicy(mode="REMOTE_DISABLED").local_files_only is True
    with pytest.raises(ValueError, match="remote_authorized"):
        ModelAccessPolicy(mode="EXPLICIT_REMOTE")
    with pytest.raises(ValueError, match="only with EXPLICIT_REMOTE"):
        ModelAccessPolicy(mode="REMOTE_DISABLED", remote_authorized=True)
    assert ModelAccessPolicy(
        mode="EXPLICIT_REMOTE", remote_authorized=True
    ).local_files_only is False


def test_jina_defaults_to_local_cache_only(monkeypatch):
    from ocr_core.engines.jina_vlm import JinaVLMEngine

    processor_factory, model_factory = _fake_jina_modules(monkeypatch)
    JinaVLMEngine()._ensure_loaded()

    assert processor_factory.call_args.kwargs["local_files_only"] is True
    assert model_factory.call_args.kwargs["local_files_only"] is True


def test_jina_remote_fetch_requires_explicit_authorization(monkeypatch):
    from ocr_core.engines.jina_vlm import JinaVLMEngine

    processor_factory, model_factory = _fake_jina_modules(monkeypatch)
    policy = ModelAccessPolicy(mode="EXPLICIT_REMOTE", remote_authorized=True)
    JinaVLMEngine(access_policy=policy)._ensure_loaded()

    assert processor_factory.call_args.kwargs["local_files_only"] is False
    assert model_factory.call_args.kwargs["local_files_only"] is False


def test_jina_cache_miss_does_not_retry_with_remote_access(monkeypatch):
    from ocr_core.engines.jina_vlm import JinaVLMEngine

    processor_factory, model_factory = _fake_jina_modules(
        monkeypatch, processor_side_effect=OSError("synthetic local cache miss")
    )
    engine = JinaVLMEngine()
    with pytest.raises(OSError, match="synthetic local cache miss"):
        engine._ensure_loaded()

    assert processor_factory.call_count == 1
    assert processor_factory.call_args.kwargs["local_files_only"] is True
    model_factory.assert_not_called()


def _fake_hub(monkeypatch, result, side_effect=None):
    module = types.ModuleType("huggingface_hub")
    download = Mock(return_value=result)
    if side_effect is not None:
        download.side_effect = side_effect
    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return download


def test_onnx_model_resolution_is_local_only_by_default(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"synthetic model placeholder")
    download = _fake_hub(monkeypatch, result=str(model_path))
    detector = OnnxWordDetector(cache_dir=tmp_path)
    detector._download_model()
    assert download.call_args.kwargs["local_files_only"] is True


def test_onnx_remote_resolution_requires_explicit_authorization(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"synthetic model placeholder")
    download = _fake_hub(monkeypatch, result=str(model_path))
    policy = ModelAccessPolicy(mode="EXPLICIT_REMOTE", remote_authorized=True)
    OnnxWordDetector(cache_dir=tmp_path, access_policy=policy)._download_model()
    assert download.call_args.kwargs["local_files_only"] is False


def test_onnx_local_cache_miss_is_not_retried_remotely(monkeypatch, tmp_path):
    download = _fake_hub(
        monkeypatch, result=str(tmp_path / "missing.onnx"),
        side_effect=RuntimeError("synthetic cache miss")
    )
    detector = OnnxWordDetector(cache_dir=tmp_path)
    with pytest.raises(LocalModelUnavailableError, match="local cache"):
        detector._download_model()
    assert download.call_count == 1
    assert download.call_args.kwargs["local_files_only"] is True


def test_auto_detector_falls_back_to_projection_without_network(monkeypatch):
    fake_ort = types.ModuleType("onnxruntime")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    detector = create_detector("auto")
    assert isinstance(detector, OnnxWordDetector)
    assert detector.fallback_to_projection is True

    def cache_miss():
        raise LocalModelUnavailableError("synthetic cache miss")

    monkeypatch.setattr(detector, "_download_model", cache_miss)
    image = Image.new("L", (120, 60), color=255)
    ImageDraw.Draw(image).rectangle([20, 10, 90, 45], fill=0)
    boxes = detector.detect(image)

    assert boxes
    assert all(box.source == "projection" for box in boxes)


def test_core_import_and_auto_fallback_need_no_hf_or_network(monkeypatch):
    blocked = {"torch", "transformers", "huggingface_hub", "onnxruntime"}
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(f"blocked optional dependency: {name}")
        return real_import(name, *args, **kwargs)

    def forbidden_network(*args, **kwargs):
        raise AssertionError("network access attempted during offline core import")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)

    from ocr_core.engines import jina_vlm
    from ocr_core.preprocess.word_detector import create_detector

    assert jina_vlm.JinaVLMEngine().access_policy.local_files_only is True
    assert jina_vlm.is_available() is False
    assert isinstance(create_detector("auto"), ProjectionDetector)
