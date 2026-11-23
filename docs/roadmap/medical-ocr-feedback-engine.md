# Medical OCR Feedback Workflow

**Status:** Proposal only — no OCR behavior is changed by this document.

## Objective

Capture human corrections to OCR output in a structured, auditable way while preserving the existing OCR engine contracts, visual-evidence rules, telemetry, and benchmark thresholds.

## Proposed correction record

Each correction should contain:
- document-local opaque reference (not patient name or raw identifier);
- page and region coordinates where available;
- language/script and document class;
- engine and engine version;
- original OCR text and corrected text, stored only in an authorized local workspace;
- correction category (recognition, segmentation, RTL/order, normalization, terminology, field extraction);
- reviewer confirmation state;
- source/evidence reference;
- timestamp and schema version.

Do not place patient-identifiable text in public CI logs, GitHub issues, telemetry, or cloud services. Public test fixtures must be synthetic or properly de-identified.

## Workflow

1. OCR produces a result with the existing confidence semantics; unknown confidence remains unknown.
2. User selects a region and supplies a correction.
3. The UI records the correction as a candidate, not an automatically accepted global rule.
4. A local review step classifies the correction and checks for conflicting terminology.
5. Accepted generic corrections may be exported to a reviewed correction set.
6. Regression evaluation compares the prior and proposed correction sets.
7. Promotion requires human review and benchmark evidence.

## Evaluation

Measure at minimum:
- CER and WER by language and document class;
- field-level extraction accuracy where labels exist;
- false correction rate;
- regressions on previously passing samples;
- performance and memory impact.

Separate recognition errors from preprocessing, layout, RTL ordering, post-processing, and glossary errors. Do not claim improvement from a small or non-representative sample.

## Safety and compatibility

- Preserve the existing visual evidence charter and evidence boundaries.
- Do not infer correctness from model confidence alone.
- Keep engine failures represented through the established result/error contract.
- Do not alter source images or original OCR artifacts.
- Keep optional dependencies optional.
- No automatic retraining or cloud transmission.
- Corrections affecting medical meaning require qualified human review.

## Acceptance criteria for a future implementation

- schema validation and migration tests;
- tests for contradictory and duplicate corrections;
- tests for synthetic patient-data leakage into logs;
- CER/WER before-and-after report;
- negative tests proving unreviewed corrections are not promoted;
- reproducible benchmark fixture and rollback to prior correction set.

## Integration boundary

`omni-medical-suite` should consume the shared `ocr-core` implementation rather than reimplementing correction storage, OCR metrics, or engine routing. Coordinate any integration with existing open PRs before implementation.
