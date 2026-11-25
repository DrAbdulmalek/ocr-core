# OCR Result Contract (schema 1.0)

Integrated roadmap priority 8. The unified OCR result (`OCRResult`) is an
explicit, versioned JSON Schema. Every engine — in-process or external —
emits one structure; ensemble voting, layout analysis and the indexer
(with per-word bboxes) read it without adapters.

Schema file: [`src/ocr_core/schemas/ocr_result.schema.json`](../src/ocr_core/schemas/ocr_result.schema.json)
(JSON Schema draft-07). Load it in Python:

```python
from ocr_core.schemas import load_schema
schema = load_schema()  # draft-07 dict, schema_version pinned to "1.0"
```

## Origin

Before this contract was promoted, the result lived only as a Python
dataclass (`src/ocr_core/engines/base.py`, `OCRResult`: `text`, `engine`,
`confidence`, `processing_time`, `error`, `pages`, `meta`) plus ad-hoc
per-engine dicts (EasyOCR/Paddle line records, the downstream dict built
in `src/ocr_core/commands/builtins/ocr.py`). The schema mirrors the
dataclass fields exactly; anything not present in code today is a
**target contract** field that producers add when emitting schema-1.0
documents.

## Required fields

| Field | Type | Required | Origin | Notes |
|---|---|---|---|---|
| `schema_version` | string, const `"1.0"` | yes | target contract | breaking changes bump the major version |
| `engine` | string | yes | in code (`OCRResult.engine`) | ensemble emits `ensemble:<winner>`; may be `""` on early failure |
| `language` | string | yes | target contract | e.g. `ar` / `en`; engines today carry `ara+eng` in `meta.lang` |
| `text` | string | yes | in code | full recognized text |
| `words[]` | array of `$defs/word` | yes | target contract | each word requires `text`, `confidence`, `bbox` |
| `page` | integer >= 0 | yes | target contract | zero-based page index; legacy `pages` is a count, not an index |
| `processing_ms` | number >= 0 | yes | target contract | legacy `processing_time` is seconds |
| `sidecar` | object | optional | target contract | auxiliary payload (e.g. per-engine documents) |
| `confidence` | number 0-1 | optional | in code | page-level; see policy below |
| `error` | string or null | optional | in code | failure reported, never raised |
| `pages`, `processing_time`, `meta` | various | optional | in code | legacy / auxiliary fields |

Top-level `additionalProperties: true`: engines may carry extras (e.g.
`meta`, diagnostics) without breaking validation.

## Confidence policy

The scale is **0-1 everywhere** (page-level and per word), enforced by
`minimum: 0` / `maximum: 1`. It matches the raw scores of EasyOCR and
PaddleOCR. `0.0` means *unknown* — it is never fabricated
(`src/ocr_core/engines/base.py` policy). Tesseract reports `0.0`.
Consumers (and the charter's R15 uncertain-marking) must not treat
unknown as low confidence: `commands/builtins/ocr.py` passes `1.0` to the
charter processor when confidence is unknown.

## Geometry

`words[].bbox` is `[x, y, w, h]` in pixels, integers, top-left origin.
Producers must convert: EasyOCR/Paddle emit 4-point polygons
(`[[x, y], ...]`), `preprocess.word_detector` emits xyxy — both become
xywh ints (`regional.normalize_bbox` handles the conversion). Width and
height are non-negative (`layout_analysis._validate_bbox` convention).

## How consumers use it

- **Ensemble voting** (`src/ocr_core/engines/ensemble.py`): ranks
  candidate results by `confidence x length x validity`, keeps
  unknown-confidence (`0.0`) engines in a separate tier ranked by
  `length x validity` — a `0.0` is never upgraded. The per-engine
  breakdown (`confidence`, `error`, `chars`) in `meta.per_engine` maps
  1:1 into `sidecar` entries keyed by engine name.
- **Manyhands-style disagreement views**: each engine's full result
  document (or its `words[]`) travels in `sidecar` under its engine name;
  disagreement is computed by comparing word bboxes and text across
  entries. Numeric confidence MUST NOT resolve a geometry conflict by
  itself (`docs/reanalysis-lifecycle-contract.md`).
- **Layout / indexer with per-word bboxes**: read `words[].bbox`
  directly — EasyOCR polygon arrays, Paddle row dicts and
  `layout_analysis` hypotheses all normalize to the same xywh ints, so no
  per-engine adapters are needed downstream.

## GPL/AGPL engines — license firewall

GPL/AGPL-licensed engines (and any other non-permissive engine) MUST NOT
be imported into or linked with ocr-core processes. They participate only
by emitting a schema-1.0 JSON document through **file exchange**: the
engine writes the OCRResult JSON to disk (or stdout) and ocr-core reads
and validates it against this schema. This keeps derivative-work
boundaries clean (the same cleanroom principle as the ABBYY pipeline)
while the unified structure makes the consumer path identical for every
engine.

## Validation

`tests/test_ocr_result_schema.py` asserts the schema loads, is valid
draft-07 (structural checks always; `jsonschema` checks when the library
is installed), a valid sample passes, and a sample missing
`words[].bbox` fails. The dataclass fields of `OCRResult` must stay
covered by the schema's properties.

`tests/test_contract.py` (the **contract gate**, issue #73) additionally
pins the whole surface across versions via a hashed golden snapshot in
[`tests/golden_contract_signature.json`](../tests/golden_contract_signature.json):
dataclass field signatures, schema version and required fields, the
canonical example below, the confidence/bbox semantics and the MIT
license. The golden file is self-verifying (sha256 over its own payload).

## Canonical example

The canonical schema-1.0 document — pinned as the golden signature and
kept identical to `VALID_RESULT` in `tests/test_ocr_result_schema.py`
(two sources of truth are not allowed: they drift, or they are one):

```json
{
  "schema_version": "1.0",
  "engine": "ensemble:tesseract",
  "language": "ar",
  "text": "نص تجريبي 123",
  "confidence": 0.87,
  "words": [
    {"text": "نص", "confidence": 0.9, "bbox": [10, 20, 30, 12]},
    {"text": "تجريبي", "confidence": 0.84, "bbox": [45, 20, 52, 12]},
    {"text": "123", "confidence": 0.0, "bbox": [100, 21, 20, 11]}
  ],
  "page": 0,
  "processing_ms": 123.4,
  "error": null,
  "pages": 1,
  "processing_time": 0.1234,
  "meta": {"lang": "ara+eng", "psm": "6"},
  "sidecar": {"tesseract": {"words": [], "confidence": 0.0}},
  "engine_extra": {"custom": true}
}
```

## Change policy / سياسة التغيير

The **contract surface** is everything that defines what producers emit
and what consumers read: `src/ocr_core/schemas/`, `docs/CONTRACT.md`,
`src/ocr_core/engines/base.py` (`OCRResult`), the golden signature
(`tests/golden_contract_signature.json`, `tests/test_contract.py`).

Gate rules (issue #73, decided 2026-11-25 — enforced mechanically by
`.github/workflows/contract-guard.yml` + `scripts/ci/contract_guard.py`):

1. **Any change to the contract surface requires a MAJOR package version
   bump** (`0.7.0 → 1.0.0`, then `1.x.y → (x+1).0.0`). A PR touching any
   of these files without a major bump fails CI.
2. **Breaking changes** (removing/renaming a field, changing a type,
   adding a required field, tightening a constraint) additionally bump
   the schema major (`schema_version` `"1.0"` → `"2.0"`) and require
   **advance notice**: an issue opened *before* merge, linking the three
   consumer repos below, left open until each confirms a migration path.
3. The golden signature is updated **in the same PR** as the change, with
   a one-line justification in the PR description.
4. Deprecations are additive first: a field is removed only in a later
   major release that follows rule 2.

## Consumers

The contract exists because these repos depend on `OCRResult` (the
consolidation plan's "core + consumers" architecture; consumer
registration tracked in the merge-plan issues):

| Repo | Reads | Tracking |
|---|---|---|
| [`arabic-handwriting-ocr-toolkit`](https://github.com/DrAbdulmalek/arabic-handwriting-ocr-toolkit) | engine results + corrections dataset | migration issue #1 (AHT-01) |
| [`arabic-medical-handwriting`](https://github.com/DrAbdulmalek/arabic-medical-handwriting) | engine results | issue #7 (AMH-01) |
| [`medical-rag-ar`](https://github.com/DrAbdulmalek/medical-rag-ar) | engine results before release | compatibility issue #5 (MRA-01) |

`common-core`, when seeded (RST-02), must consume this contract — never
copy it.
