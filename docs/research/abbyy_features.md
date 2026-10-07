# ABBYY features — clean-room extraction (legal source: public documentation)

**Source of truth**: publicly available product documentation only
(help.abbyy.com / docs.abbyy.com — concepts, stage names, mode names).
**Rule**: we extract the *idea*, never the code.

## The golden rule

| Allowed (idea) | Forbidden (code/artifacts) |
|---|---|
| Six-stage pipeline structure | `.rom` / `.dat` files, weights, dictionaries |
| Layout-analysis mode names & semantics | Compiled recognition algorithms |
| Complex-table support as a goal | Neural networks, pattern-training models |
| Selective re-recognition concept | Any content behind the vendor EULA |
| RTL/Arabic handling requirements | Anything not in public documentation |
| Publicly declared quality metrics (CER/WER) | Unverified accuracy claims |

If a fact is found in the vendor's public documentation, the *idea* is
extractable. If it lives inside an executable, a model or a data file, it is
protected. Nothing from binaries was consulted for this work.

## 1. Six-stage pipeline

Public-documentation concept:

```
Page image
→ Preprocessing (deskew, denoise, contrast)
→ Layout analysis (text, tables, pictures, headers)
→ Recognition
→ Page synthesis (rebuild page structure)
→ Document synthesis (assemble pages)
→ Export (PDF, DOCX, MD, JSON)
```

**Local implementation**: `src/ocr_core/pipeline.py`
(`DocumentPipeline`, `PipelineStage`).

## 2. Layout-analysis modes

Public-documentation concept: `Simple` (text only), `Complex`
(text + tables + pictures), `Table-only`, `Automatic`.

**Local implementation**: `ocr_core.pipeline.LayoutMode`
(`simple`, `complex`, `tables`, `auto`) — `AUTO` picks the effective mode
from the region kinds the layout provider emitted.

## 3. Selective re-recognition

Public-documentation concept: select a region, adjust its properties,
re-run recognition on that region only.

**Local implementation**: `src/ocr_core/regional.py`
(`extract_region`, `rerun_region`, `rerun_regions`) with both `xywh`
(ui/LayoutDocument convention) and `xyxy` (bbox-editor convention) formats.
Engine confidence stays "unknown = 0.0" per repo policy; multi-region
aggregates are never averaged into a fabricated score.

## 4. Arabic / RTL support requirements

Public-documentation concept: complex scripts need dedicated handling —
RTL direction, letter shaping, diacritics.

**Local implementation**: the existing `ocr_core.rtl_utils` fixer is exposed
as a pipeline hook through `ocr_core.postprocess.arabic_rtl`
(`ArabicRTLPostProcessor`). No new normalization rules were invented; the
wrapper delegates verbatim so the two entry points cannot drift.

## 5. Open-source replacements mapped to the same features

| Feature | Open-source vehicle | Status in ocr-core |
|---|---|---|
| Layout analysis | Surya / deterministic projection analyzer | `layout_baseline.py` (projection) — Surya is a documented future provider |
| Table recognition | Surya tables / OpenDataLoader PDF | future provider hook (`layout_provider`) |
| Arabic OCR | Tesseract / EasyOCR / PaddleOCR | `engines/` shipped |
| Batch processing | pipeline over pages | `DocumentPipeline.process_pdf` |
| Quality metrics | CER/WER harness | `benchmarks/` (golden + ci_suite) |

## 6. Legal notes (verbatim policy)

- No vendor binaries were downloaded, decompiled, unpacked or analyzed.
- No vendor model/dictionary artifacts are present in this repository.
- Any future contributor must keep this clean-room boundary: concepts from
  public docs only; when in doubt, leave it out and record the question here.

---

## D-004: ABBYY structure extraction (clean room)

**Reference**: ABBYY FineReader product documentation (public).
**Extracted**: six-stage pipeline structure + layout-analysis modes +
selective re-recognition concept + Arabic/RTL handling requirements.
**Implemented in**: `ocr_core.pipeline` + `ocr_core.regional` +
`ocr_core.postprocess.arabic_rtl`.
**Legal note**: no proprietary code, models or dictionaries were used or
consulted; ideas only, from public documentation.
**Tests**: `tests/test_pipeline.py`, `tests/test_regional.py`,
`tests/test_postprocess_arabic_rtl.py`.
