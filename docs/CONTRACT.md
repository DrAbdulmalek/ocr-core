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
