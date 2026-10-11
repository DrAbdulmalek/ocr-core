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


## Remediation specification — not yet implemented

The diagram above is the target, not a claim that every stage exists. Do not export or publish until all gates below exist and are covered by tests.

1. **Content identity:** persist a versioned canonical serialization hash (SHA-256) over the annotation payload, including verbatim text, normalized text, region/box metadata, source reference, schema version and applicable correction metadata. Keep raw source hashes separate from derived-record hashes. A repeated `(source_id, source_revision, content_hash)` must be idempotent; a changed hash must create a new revision, not overwrite history.
2. **Approval binding:** store approval actor, timestamp, consent/purpose, decision, policy version and the exact content hash approved. Any content mutation invalidates that approval and returns the record to pending review. Fail closed for absent or mismatched approvals.
3. **Export ledger:** append-only records with export_id, dataset/destination identifier, selected record IDs and hashes, split assignment, exporter version, schema version, privacy-policy version, approval snapshot, creation time, result/status and remote revision/commit if available. Do not store credentials in the ledger.
4. **Destination-specific authorization:** require explicit authorization for each destination and purpose (local file, private dataset, public Hugging Face dataset, etc.). Public publish must be a separate, human-confirmed operation; no automatic publish from approval or CI.
5. **Reconciliation/rollback:** dry-run first; export to a versioned staging path; verify count/hash manifest; then promote only with authorization. Record revocations and tombstones rather than silently deleting history. Public dataset corrections require a destination-specific remediation plan.
6. **Validation:** test changed content invalidates approval; same content re-ingestion is idempotent; rejected/pending records never export; tampered hashes fail closed; destination mismatch is denied; retry does not duplicate; rollback restores previous pointer; no secrets/PII in logs; split assignment remains deterministic and leakage-safe.

### Canonical-module decision
The three duplicated modules in `src/ocr/` and `hf-space/src/ocr/` are not deleted or moved in this audit. Before canonicalization, determine packaging/deployment constraints for the HF Space, choose one source of truth, and add an import/package test for both deployment targets. A sync script is acceptable only if it verifies byte-for-byte parity and fails CI on drift; it must not silently overwrite divergent code.
