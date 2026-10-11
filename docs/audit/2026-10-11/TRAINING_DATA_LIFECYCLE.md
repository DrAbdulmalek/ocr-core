# Training-data lifecycle audit (as actually implemented)

```
OCR editor → human correction → explicit approval → training record
  → deterministic id + content hash → privacy gate → dataset validation
  → export ledger → optional dataset export
```

| Stage | Implemented? | Where | Notes |
|---|---|---|---|
| OCR editor / slice capture | yes (PR #166) | `src/ocr/slice_store.py` | identical twin in `hf-space/src/ocr/` |
| Pattern learning (ABBYY-style templates) | yes (PR #166) | `src/ocr/pattern_store.py` | 96×48 bitmap b64 |
| Human approval gate | yes (PR #166) | `scripts/trainingdb_sync.py::validate_privacy_approvals` | **per-sample, fail-closed**; a legacy batch `human_reviewed: true` flag is rejected — covered by `tests/test_trainingdb_privacy_gate.py` |
| Deterministic id + content hash | **PARTIAL** | split is `sha256(slice_id)` | ids exist; there is **no stored content hash** of the annotation, so re-ingestion cannot be proven idempotent from the record itself |
| Dataset validation | PARTIAL | `training_export.py` | validates counts/splits, not schema/privacy |
| Export ledger | **NO** | — | nothing records *what/when/under which policy/from which source revision* |
| Per-destination export authorisation | **NO** | — | no explicit gate before a remote publish |
| Rejected-record exclusion | yes | gate is fail-closed | unapproved records are not exported |
| Uncertainty not silently promoted | yes (ocr-core) | `build_training_record` | promotes only with confidence + marker/legend + source_reference + `evidence_confirmed` |

## Divergent implementations
`omni-ocr-training-db` is a **data-only** store (no schema/migration code). The only lifecycle code lives
in `omni-medical-suite` PR #166 — **duplicated verbatim** in `src/ocr/` and `hf-space/src/ocr/`
(sha256 `0d0f45db142ea258`, `a6e86f97c2312d0a`, `d683f83d77a9452f`). One canonical module + documented
contract is the minimum justified fix; not performed here (destructive to an open PR's layout).

## Privacy
Current content: 3 seed slices of Telegram **screenshot UI text** ("Chats", "Channels") —
no clinical data, and the repository is private. The risk is structural (missing ledger), not present content.
No dataset was written, exported, or published during this audit.
