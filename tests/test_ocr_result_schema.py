"""Schema contract tests (roadmap P8): ocr_result.schema.json, draft-07.

The JSON Schema is kept consistent with the ``OCRResult`` dataclass in
``ocr_core.engines.base`` — structural checks run without any third-party
library; ``jsonschema``-based checks activate only when it is installed.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from ocr_core.schemas import SCHEMA_VERSION, load_schema

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

REQUIRED_TOP = ["schema_version", "engine", "language", "text", "words",
                "page", "processing_ms"]
REQUIRED_WORD = ["text", "confidence", "bbox"]

VALID_RESULT = {
    "schema_version": "1.0",
    "engine": "ensemble:tesseract",
    "language": "ar",
    "text": "نص تجريبي 123",
    "confidence": 0.87,
    "words": [
        {"text": "نص", "confidence": 0.9, "bbox": [10, 20, 30, 12]},
        {"text": "تجريبي", "confidence": 0.84, "bbox": [45, 20, 52, 12]},
        {"text": "123", "confidence": 0.0, "bbox": [100, 21, 20, 11]},
    ],
    "page": 0,
    "processing_ms": 123.4,
    "error": None,
    "pages": 1,
    "processing_time": 0.1234,
    "meta": {"lang": "ara+eng", "psm": "6"},
    "sidecar": {
        "tesseract": {"words": [], "confidence": 0.0},
    },
    "engine_extra": {"custom": True},
}


def _schema() -> dict:
    return load_schema()


# -- loading ----------------------------------------------------------------

def test_load_schema_returns_dict():
    schema = _schema()
    assert isinstance(schema, dict)
    assert schema["type"] == "object"


def test_load_schema_is_repeatable_and_independent():
    a, b = load_schema(), load_schema()
    assert a == b and a is not b


def test_schema_version_constant_matches():
    schema = _schema()
    assert SCHEMA_VERSION == "1.0"
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


# -- structural draft-07 checks (no third-party library needed) -------------

def test_schema_declares_draft07():
    assert _schema()["$schema"] == "http://json-schema.org/draft-07/schema#"


def test_schema_required_fields():
    schema = _schema()
    for field in REQUIRED_TOP:
        assert field in schema["required"], f"missing required field {field}"
    # sidecar is intentionally optional
    assert "sidecar" in schema["properties"]
    assert "sidecar" not in schema["required"]


def test_schema_additional_properties_true_at_page_level():
    assert _schema()["additionalProperties"] is True


def test_schema_word_defs():
    schema = _schema()
    word = schema["$defs"]["word"]
    assert word["type"] == "object"
    for field in REQUIRED_WORD:
        assert field in word["required"], f"missing required word field {field}"
    assert schema["properties"]["words"]["items"]["$ref"] == "#/$defs/word"


def test_schema_confidence_bounds_are_0_to_1():
    schema = _schema()
    for pointer in (schema["properties"]["confidence"],
                    schema["$defs"]["word"]["properties"]["confidence"]):
        assert pointer["type"] == "number"
        assert pointer["minimum"] == 0
        assert pointer["maximum"] == 1


def test_schema_bbox_is_four_nonnegative_ints():
    bbox = _schema()["$defs"]["word"]["properties"]["bbox"]
    assert bbox["minItems"] == bbox["maxItems"] == 4
    assert bbox["items"] == {"type": "integer", "minimum": 0}


def test_schema_stays_consistent_with_ocrresult_dataclass():
    # The schema mirrors the in-code dataclass: every OCRResult field must
    # remain a declared property so adapters never become necessary again.
    from ocr_core.engines.base import OCRResult

    schema = _schema()
    for f in dataclasses.fields(OCRResult):
        assert f.name in schema["properties"], (
            f"OCRResult.{f.name} is not declared in ocr_result.schema.json"
        )


# -- jsonschema-backed checks (skipped when the library is absent) ----------

@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_schema_is_valid_draft07():
    jsonschema.Draft7Validator.check_schema(_schema())


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_valid_sample_passes_validation():
    validator = jsonschema.Draft7Validator(_schema())
    errors = list(validator.iter_errors(VALID_RESULT))
    assert errors == [], [e.message for e in errors]


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_missing_word_bbox_fails_validation():
    bad = json.loads(json.dumps(VALID_RESULT))
    del bad["words"][0]["bbox"]
    validator = jsonschema.Draft7Validator(_schema())
    errors = list(validator.iter_errors(bad))
    assert errors, "missing words[].bbox must not validate"
    assert any("bbox" in e.message for e in errors)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_confidence_out_of_range_fails_validation():
    bad = json.loads(json.dumps(VALID_RESULT))
    bad["words"][0]["confidence"] = 1.5  # 0-100-style score must be rejected
    validator = jsonschema.Draft7Validator(_schema())
    assert list(validator.iter_errors(bad))
    bad2 = json.loads(json.dumps(VALID_RESULT))
    bad2["words"][1]["bbox"] = [10, 20, 30, -1]  # negative extent
    assert list(validator.iter_errors(bad2))


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_wrong_schema_version_fails_validation():
    bad = json.loads(json.dumps(VALID_RESULT))
    bad["schema_version"] = "2.0"
    validator = jsonschema.Draft7Validator(_schema())
    assert list(validator.iter_errors(bad))
