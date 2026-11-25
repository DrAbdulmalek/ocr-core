# OCR Core consumer compatibility snapshot

Audit date: 2026-11-26. This is a source-level compatibility snapshot, not a complete inventory of every private or unindexed repository.

## Confirmed direct consumer

### IntelliFile Manager

Evidence inspected on its default branch:

- `requirements-ocr-core.txt` pins `marathon-ocr-core[tesseract]` to commit `5613a1c05f7b657678a810b96c6ba5cab3b92e53` (the v0.5.0 release commit), rather than tracking a moving branch.
- `src/core/ocr_service.py` imports `PaddleEngine` and `TesseractEngine`, calls `available()` and `process_image()`, and consumes `OCRResult.text`, `engine`, `confidence`, and `error`.
- The same service optionally imports `ocr_core.postprocess.corrections_ar`, `ocr_core.postprocess.normalization`, and `ocr_core.preprocess.enhance`.
- Its adapter deliberately preserves the existing dictionary contract: `extracted_text`, `ocr_engine`, `ocr_success`, optional `ocr_confidence`, and diagnostic error keys.

Source references:
- https://github.com/DrAbdulmalek/intelli-file-manager/blob/82e3e49196df1d1d2bc3e5b3a4fc81c6e739cc82/requirements-ocr-core.txt
- https://github.com/DrAbdulmalek/intelli-file-manager/blob/82e3e49196df1d1d2bc3e5b3a4fc81c6e739cc82/src/core/ocr_service.py
- https://github.com/DrAbdulmalek/intelli-file-manager/blob/82e3e49196df1d1d2bc3e5b3a4fc81c6e739cc82/tests/unit/test_ocr_service.py

Evidence status: **Proven** for the inspected files and pinned SHA; **Partially Proven** for end-to-end compatibility with the proposed ModelStore integration because IntelliFile has not yet been updated to configure `model_store_root` or a model revision.

## Other searched repositories

A default-branch code search for `ocr_core` returned no matches in the inspected repositories:
- `DrAbdulmalek/omni-medical-suite`
- `DrAbdulmalek/medical-rag-ar`
- `DrAbdulmalek/arabic-medical-handwriting`
- `DrAbdulmalek/Faris-DICOM-Studio`

Evidence status: **Unproven** as a claim of no dependency. GitHub code search may omit unindexed repositories, private inaccessible sources, vendored code, or non-default branches. This is not a basis for declaring those repositories independent.

## Compatibility and rollout rules

1. Do not change the locked `OCRResult` public contract or serialized schema in this integration.
2. IntelliFile's existing pin remains unchanged. Updating it is a separate consumer PR after the owner reviews and merges the relevant OCR Core PRs.
3. Do not silently enable ModelStore or remote access in IntelliFile. A consumer must explicitly configure a local store root and immutable model revision; any remote policy remains a separate explicit owner decision.
4. Validate consumer compatibility against the exact OCR Core commit before changing its pin. Preserve the legacy result dictionary and the sensitive-file barrier.
5. No real medical images, OCR text, or patient data were used in this audit.

## Remaining validation

- Build and test IntelliFile against the eventual merged OCR Core SHA in a separate PR.
- Verify the actual local model manifest and license/provenance for each model before use.
- Run private, owner-controlled Arabic medical CER/WER and layout benchmarks. No clinical accuracy claim is made here.
