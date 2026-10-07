# Layout Re-analysis Lifecycle and Conflict Policy Contract

## Purpose
This contract defines how a persisted LayoutDocument behaves when new layout-analysis runs occur after previous analysis, review, or explicit apply operations.

## 1. Lifecycle boundary
Analysis, hypothesis generation, matching, review, explicit apply, and later re-analysis are distinct events. A new analysis run is a new observation and MUST NOT replace the canonical document.

## 2. Identity authority
MedicalRegionData.region_id remains the sole persistent identity. analysis_run_id, hypothesis IDs, analyzer/configuration identifiers, and provenance identifiers MUST NOT replace it.

## 3. Lifecycle concepts
Minimum conceptual states:
- INFERRED: analysis/import observation not explicitly accepted.
- REVIEWED: explicitly reviewed but not necessarily applied.
- APPLIED: explicit decision incorporated the proposal.
- MANUAL: explicitly created or manually modified.
- CONFLICTED: later evidence cannot be safely reconciled without explicit resolution.
- SUPERSEDED: an older observation is no longer the active observation; this does NOT delete a persistent region.
These are lifecycle/provenance concepts, not OCR confidence states. Existing compatibility fields MUST NOT be silently reinterpreted.

## 4. Re-analysis is additive
Later analysis MUST be treated as a separate analysis event. It MUST NOT replace all regions, delete omitted regions, reset manual state, replace persistent IDs, or silently overwrite applied fields. Cross-run matching and explicit apply remain required.

## 5. Applied and manual regions
If a later hypothesis corresponds to an existing persistent region, matching MUST be explicit and the persistent identity MUST survive. Differences become explicit update proposals; protected/manual differences remain conflicts unless explicitly resolved.
Manual creation/modification has highest protection priority. Later analysis MUST NOT implicitly overwrite geometry, type, content, or manual provenance.

## 6. Provenance on apply
When an inferred/imported hypothesis is explicitly applied, the analysis run and hypothesis remain historical provenance and the explicit acceptance/application MUST be auditable. Machine application MUST NOT itself imply manual status. Human-authored changes require an explicit manual-modification event.

## 7. Stale state
Analysis, matching results, and apply decisions MUST be bound to the relevant source/document identity and expected document state. If the canonical document changes, stale apply MUST fail unless an explicit compatibility policy validates it. The document MUST remain unchanged on stale failure.

## 8. Superseding analysis
A newer run MAY supersede an older observation for audit purposes. Superseding MUST NOT delete the persistent region. Only explicit apply can change the canonical document.

## 9. Conflict categories
At minimum distinguish geometry disagreement, region-type disagreement, hypothesis disappearance, new overlap, source/document mismatch, stale document state, and manual-protection conflict. Numeric confidence MUST NOT resolve a conflict by itself.

## 10. No implicit deletion
Absence from a later run, analysis failure, configuration change, UNMATCHED_EXISTING, or superseding a run MUST NOT delete/deactivate a persistent region. Deletion/deactivation requires a separately defined explicit policy.

## 11. Determinism and audit
For identical inputs, provenance, policy, and document state, lifecycle results MUST have deterministic ordering and serialization. Audit records SHOULD identify the analysis run, hypothesis, persistent region, matching/review decision, apply decision, changed fields, protection state, expected document state, and governing policy/version.

## 12. Serialization/versioning
Lifecycle metadata MUST be versioned independently from layout_version, OCR confidence calibration, analyzer version, and matching policy. Existing payloads MUST remain readable unless an explicit migration is introduced.

## 13. Safety invariant
new analysis != document replacement; new hypothesis != persistent region; absence != deletion; confidence != authorization; matching != mutation. The only canonical mutation boundary remains explicit apply.

## 14. Required future runtime tests
1. Later analysis does not mutate the document.
2. Persistent identity survives re-analysis.
3. Manual regions remain protected.
4. Omitted regions are not deleted.
5. Overlapping hypotheses do not auto-create duplicates.
6. Stale analysis/apply is rejected.
7. Superseded runs preserve historical provenance.
8. Geometry/type conflicts remain explicit.
9. Source/document mismatch is rejected.
10. Lifecycle serialization is deterministic.
11. Legacy payloads remain readable.
12. Unrelated regions remain unchanged.

## 15. Non-goals
No production ML analyzer, automatic semantic matching, automatic conflict resolution, deletion policy, OCR execution, SQLite/HF/React integration, dataset export, or destructive replace-all workflow.

## Architectural principle
Analysis Run → Explicit Cross-Run Matching → Review/Decision → Explicit Apply → Canonical LayoutDocument → Later Analysis. Every cycle starts a new observation; no analysis run acquires implicit authority over the canonical document.