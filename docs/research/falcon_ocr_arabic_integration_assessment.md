# Falcon-OCR-Arabic Integration Assessment

**Status:** RESEARCH ONLY — not implemented, not benchmarked locally, and not approved for merge.  
**Target:** `ocr-core` first; consumers may later include `omni-medical-suite`.  
**Date checked:** 2026-10-11.

## Decision

Evaluate Falcon-OCR-Arabic as an **optional Arabic document OCR engine** behind the existing `ocr-core` engine interface. Do not replace the default router, existing engines, or the canonical OCR result contract on the basis of a vendor benchmark alone.

The intended deployment is local/offline. The hosted playground is useful only for non-sensitive demonstrations; do not upload patient records, prescriptions, identity documents, or other confidential material to it.

## Upstream evidence

- TII announcement: https://falcon-lm.github.io/blog/falcon-ocr/
- TII Arabic announcement: https://falcon-lm.github.io/ar/blog/falcon-ocr/
- TII-hosted demo: https://huggingface.co/spaces/tiiuae/falcon-ocr-arabic
- TII model collection: https://huggingface.co/tiiuae

The announcement reports a 270M-parameter Arabic-adapted model, 81.9% text accuracy on the authors' 11,974-sample / 15-category benchmark, and Table TEDS 59.95%. It reports strong results on official documents, administrative forms, receipts, and invoices, and weaker relative results on newspapers, magazines, comics, and handwriting.

These are **upstream-reported results**, not results independently reproduced by this repository. The metric is the authors' normalized edit-distance-derived page text accuracy; it must not be treated as equivalent to CER, WER, or clinical correctness.

At the time of this assessment, the official demo is discoverable, but a directly usable public Arabic checkpoint, exact inference API, model-weight license, revision pin, and offline-loading instructions have **not been verified**. The existing `tiiuae/Falcon-OCR` checkpoint is not proof that the Arabic-adapted weights are available under the same model ID or terms. Do not assume they are interchangeable.

## Proposed integration boundary

1. Add a separate optional adapter (proposed name: `FalconArabicEngine`) under `ocr_core.engines` only after the official checkpoint and supported inference recipe are verified.
2. Keep model loading lazy. Importing `ocr_core` or running the baseline engines must not download weights or import heavy model dependencies.
3. Use an explicit optional extra (proposed: `falcon-arabic`); do not add it to the default or `all` extras until compatibility, license, and resource requirements are proven.
4. Adapt output into the existing `OCRResult` contract. Preserve raw model output as `text_verbatim`; any normalization or medical correction must be a separate, traceable downstream view.
5. Record engine/model ID, immutable revision or artifact hash, prompt/category, runtime settings, errors, and timing in provenance. Do not infer confidence where the model does not provide a calibrated score; unknown confidence remains unknown.
6. Support explicit output categories only when the official inference implementation confirms them: plain text, HTML tables, and LaTeX formulas. Do not claim bounding-box support unless the model actually emits coordinates.
7. No automatic cloud fallback. If local weights are unavailable, return a clear unavailable/error state rather than sending documents to the hosted playground or another service.
8. Never send OCR output to training automatically. Any training-data export remains subject to the existing explicit human-confirmation and privacy gates.

## Evaluation gate before implementation

Use a fixed, consented, non-identifying local benchmark with gold transcripts and document-type labels. Include Arabic printed medical reports, prescriptions (printed and handwritten separately), lab results, RTL tables, mixed Arabic/Latin text, Western and Eastern Arabic digits, stamps/noise, receipts/forms, and difficult scans. Keep patient data out of public fixtures.

Compare against the currently supported engines in `ocr-core` under identical preprocessing and test splits:

- CER and WER, reported separately by document class and script condition.
- Exact-match and critical-field accuracy for numbers, dates, doses, units, decimal separators, and medication names.
- Table structure/content score (TEDS or an explicitly documented equivalent) where reference HTML is available.
- Latency, peak RAM/VRAM, model disk size, cold-start time, and CPU-only feasibility.
- Reproducibility, failure handling, and no-network behavior after the weights are cached.

Publish aggregate results and test manifests, not source patient documents. Establish acceptance thresholds before examining final comparison results to reduce selection bias.

## Security, privacy, and licensing gates

- Inspect model card, code, dependency tree, license for both code and weights, and any `trust_remote_code` requirements before executing upstream code.
- Pin dependencies and model revision; record hashes. Do not execute unreviewed model code on confidential documents.
- Confirm local inference can run without network access after installation and model caching.
- Verify the actual weight license and redistribution conditions independently; do not infer them from the TII organization page or from the base Falcon-OCR model.
- Do not log document contents or patient identifiers. Keep diagnostics to metadata and redacted errors.
- The hosted playground is not an approved route for sensitive clinical material unless its data handling and retention terms have been independently reviewed and approved.

## Current disposition

**RESEARCH ONLY / BLOCKED ON UPSTREAM ARTIFACT VERIFICATION.** No engine code, dependencies, model weights, router changes, or medical-product changes are included in this assessment. Next step: verify an official downloadable Arabic checkpoint and its license/runtime instructions, then implement the smallest isolated adapter and run the evaluation gate.