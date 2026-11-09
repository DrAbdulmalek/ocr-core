# Architecture and Evidence Boundaries

> Status: documentation derived from the repository's current README and
> `pyproject.toml`. This document describes the declared architecture; it is
> not a substitute for source-level review or runtime validation.

## Package identity

- Python distribution: `marathon-ocr-core`
- Import package: `ocr_core`
- Python requirement: `>=3.10`
- Core declared dependency: `pyyaml>=6.0`
- Source package root: `src/`
- Test root: `tests/`; pytest adds `src` to `pythonpath`.

## Declared component map

| Component | Responsibility documented by the project |
|---|---|
| `ocr_core.rules` | Rule/charter evaluation and text-normalization rules |
| `ocr_core.engines` | Common OCR engine/result contract and optional engine adapters |
| `ocr_core.preprocess` | Image preparation, including crop, deskew, rotation and enhancement |
| `ocr_core.postprocess` | Arabic corrections, normalization, field extraction and deduplication |
| `ocr_core.pipeline` | Multi-stage document processing and synthesis |
| `ocr_core.regional` | Selective re-recognition of a bounded region |
| `ocr_core.rtl_utils` | Arabic RTL processing utilities |
| `ocr_core.telemetry` | Structured decision/event records |
| `ocr_core.commands` | Command registry, executor, pipeline and CLI adapter |
| `ocr_core.benchmarks` | Evaluation suites and threshold/A-B tooling |

The README describes the document pipeline as:

`Preprocess → Layout → Recognition → Page Synthesis → Document Synthesis → Export`.

It also documents optional engines and features, including Tesseract, PaddleOCR,
EasyOCR, ensemble/router, word detection and a Jina OCR VLM adapter.

## Dependency and installation boundaries

The base install is intentionally small. Heavy OCR engines, image-processing
libraries, VLM dependencies, UI dependencies and benchmark tools are exposed
through optional extras in `pyproject.toml`.

Do not make an optional engine a mandatory import-time dependency. Adapter
imports should remain lazy or fail visibly through the documented result/error
contract when an optional dependency is unavailable.

The `jina` extra downloads/uses a model whose weights are documented in the
README/metadata as CC BY-NC 4.0. The Python package's MIT license does not
change the model-weight license. Commercial use must not be inferred to be
permitted.

## Data and clinical-safety boundaries

1. OCR output is machine-generated evidence, not verified clinical truth.
2. Confidence `0.0` is documented as unknown; callers must not reinterpret it
   as certainty or silently invent a confidence value.
3. Preserve raw OCR text, engine identity, page/region coordinates and
   correction provenance when downstream consumers need auditability.
4. Never silently overwrite source text with a normalization or correction.
   Keep original and normalized forms distinguishable.
5. A successful OCR call does not establish clinical correctness. Human review
   remains required for medication names, doses, units, negation, laterality,
   dates and patient identifiers.
6. Patient documents and DICOM-derived text must remain local by default.
   Any remote processing requires a separately designed, explicit and
   auditable authorization path; no implicit cloud fallback is acceptable.

## Integration contract for consumers

Consumers such as `omni-medical-suite` should depend on a released,
version-pinned distribution rather than copying implementation files. Before
replacing an existing implementation, verify:

- API and result-schema compatibility;
- preservation of bounding boxes and coordinate conventions;
- error and missing-dependency behavior;
- confidence semantics;
- language/RTL rendering behavior;
- license compatibility for code, models and datasets;
- deterministic tests using synthetic fixtures.

## Verification status

This document was prepared from the README and package metadata. It does not
claim that every documented component has been independently exercised, that
all adapters are clinically validated, or that production integrations have
been completed. Those claims require targeted source review, tests and
measured benchmark evidence.
