"""Versioned JSON Schemas for ocr-core contracts (roadmap P8).

The unified OCR result contract (``OCRResult``) is promoted from an
implicit Python dataclass to an explicit, versioned JSON Schema so every
engine and downstream consumer (ensemble voting, layout, indexer with
per-word bboxes) reads one structure without adapters.

Usage::

    from ocr_core.schemas import load_schema

    schema = load_schema()  # -> dict (draft-07, schema_version 1.0)

Loader is robust when the package is installed: it reads via
``importlib.resources`` first and falls back to the on-disk path next to
this module (source checkouts, editable installs).
"""
from __future__ import annotations

import json
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
SCHEMA_FILE = "ocr_result.schema.json"


def load_schema(name: str = SCHEMA_FILE) -> dict[str, Any]:
    """Load a bundled JSON Schema as a dict.

    Tries ``importlib.resources`` (works for wheel/zip installs), then
    falls back to the file next to this module. Raises ``FileNotFoundError``
    only if both routes fail, which would mean a broken package.

    / يحمّل مخطط JSON المرفق مع الحزمة — أولًا عبر importlib.resources ثم
    بالمسار المجاور لهذا الملف، فيعمل مثبّتًا أو من الشجرة المصدرية.
    """
    try:
        resource = files("ocr_core.schemas").joinpath(name)
        with as_file(resource) as path:
            text = Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, ModuleNotFoundError,
            AttributeError, TypeError):
        text = (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")
    schema = json.loads(text)
    if not isinstance(schema, dict):
        raise ValueError(f"schema '{name}' must decode to a JSON object")
    return schema
