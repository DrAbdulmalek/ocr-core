"""Contract gate tests (issue #73): golden signatures of the OCRResult contract.

Locks the contract surface across versions: the ``OCRResult`` dataclass
fields, the JSON schema's version and required fields, the canonical
example, the confidence/bbox semantics and the MIT license. Any deviation
must update ``tests/golden_contract_signature.json`` **in the same PR**
and bump the package major version — enforced mechanically by
``.github/workflows/contract-guard.yml`` + ``scripts/ci/contract_guard.py``.

Complements ``tests/test_ocr_result_schema.py``: that file validates the
schema's internal structure; this file pins the contract *across versions*
via a hashed golden snapshot (golden signatures per the gate definition).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from pathlib import Path

import pytest

from ocr_core.engines.base import OCRResult
from ocr_core.schemas import SCHEMA_VERSION, load_schema

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

GOLDEN_PATH = Path(__file__).parent / "golden_contract_signature.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
REPO_ROOT = Path(__file__).resolve().parent.parent


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _norm_type(tp) -> str:
    """Stable string for a dataclass annotation. With PEP 563
    (``from __future__ import annotations``) ``f.type`` is already the
    literal source string; normalize only the typing-module prefix."""
    if isinstance(tp, str):
        return tp.replace("typing.", "")
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = typing.get_args(tp)
        rest = [a for a in args if a is not type(None)]
        if len(args) - len(rest) == 1 and len(rest) == 1:
            return f"Optional[{_norm_type(rest[0])}]"
    if origin is not None:
        return getattr(origin, "__name__", str(origin))
    return getattr(tp, "__name__", str(tp))


def _field_signature() -> list[dict]:
    sig = []
    for f in dataclasses.fields(OCRResult):
        default = ("<factory>" if f.default_factory is not dataclasses.MISSING
                   else f.default)
        sig.append({"name": f.name, "type": _norm_type(f.type),
                    "default": default})
    return sig


# -- golden integrity ---------------------------------------------------------

def test_golden_file_integrity():
    """The golden file is self-verifying: its content (minus the hash) must
    reproduce the recorded sha256 — catches manual tampering/typos."""
    payload = {k: v for k, v in GOLDEN.items() if k != "sha256"}
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    assert digest == GOLDEN["sha256"], "golden signature file was modified but not re-captured"


# -- locked contract surface --------------------------------------------------

def test_dataclass_signature_locked():
    assert _field_signature() == GOLDEN["dataclass_fields"], (
        "OCRResult dataclass changed — update tests/golden_contract_signature.json "
        "in the same PR and bump the package MAJOR version (issue #73)")


def test_schema_version_locked():
    schema = load_schema()
    assert SCHEMA_VERSION == GOLDEN["schema_version"]
    assert schema["properties"]["schema_version"]["const"] == GOLDEN["schema_version"]


def test_required_fields_locked():
    schema = load_schema()
    assert set(schema["required"]) == set(GOLDEN["schema_required_top"])
    assert len(schema["required"]) == len(GOLDEN["schema_required_top"])
    word = schema["$defs"]["word"]
    assert set(word["required"]) == set(GOLDEN["schema_required_word"])
    assert len(word["required"]) == len(GOLDEN["schema_required_word"])


def test_confidence_and_bbox_semantics_pinned():
    """Confidence is 0-1 everywhere (0.0 = unknown, never fabricated) and
    bbox is 4 non-negative integers xywh — the two semantic pillars of the
    contract (docs/CONTRACT.md)."""
    schema = load_schema()
    conf = schema["properties"]["confidence"]
    wconf = schema["$defs"]["word"]["properties"]["confidence"]
    bbox = schema["$defs"]["word"]["properties"]["bbox"]
    assert (conf.get("minimum"), conf.get("maximum")) == (0, 1)
    assert (wconf.get("minimum"), wconf.get("maximum")) == (0, 1)
    assert bbox["minItems"] == bbox["maxItems"] == 4
    assert bbox["items"] == {"type": "integer", "minimum": 0}


def test_license_is_mit():
    """The gate's licensing clause: ocr-core ships an explicit MIT LICENSE
    (official consumption by the consolidation consumers requires it)."""
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.lstrip().startswith("MIT License")
    assert "Dr Abdulmalek Al-Husseini" in text


# -- canonical example ---------------------------------------------------------

def test_canonical_example_is_the_declared_sample():
    """The golden canonical example must stay identical to the repo's own
    declared sample (``VALID_RESULT`` in test_ocr_result_schema.py): two
    sources of truth are not allowed — they drift or they are one."""
    from test_ocr_result_schema import VALID_RESULT

    assert GOLDEN["canonical_example"] == VALID_RESULT


def test_canonical_example_satisfies_schema_structure():
    schema = load_schema()
    ex = GOLDEN["canonical_example"]
    for field_name in schema["required"]:
        assert field_name in ex, f"canonical example missing required field {field_name}"
    assert ex["schema_version"] == GOLDEN["schema_version"]
    conf = ex.get("confidence")
    assert conf is None or 0.0 <= conf <= 1.0
    assert 0.0 <= ex["words"][2]["confidence"] == 0.0  # unknown stays 0.0, never fabricated
    for w in ex["words"]:
        for wf in schema["$defs"]["word"]["required"]:
            assert wf in w
        assert 0.0 <= w["confidence"] <= 1.0
        bbox = w["bbox"]
        assert len(bbox) == 4 and all(isinstance(v, int) and v >= 0 for v in bbox)


def test_dataclass_roundtrip_matches_canonical_example():
    """Legacy dataclass fields in the canonical example survive a
    OCRResult(**kwargs) → asdict() round trip unchanged."""
    ex = GOLDEN["canonical_example"]
    kwargs = {f.name: ex[f.name]
              for f in dataclasses.fields(OCRResult) if f.name in ex}
    back = dataclasses.asdict(OCRResult(**kwargs))
    for name, value in kwargs.items():
        assert back[name] == value


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_canonical_example_passes_full_validation():
    validator = jsonschema.Draft7Validator(load_schema())
    errors = list(validator.iter_errors(GOLDEN["canonical_example"]))
    assert errors == [], [e.message for e in errors]
