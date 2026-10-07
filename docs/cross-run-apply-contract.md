# Cross-Run Reconciliation Apply Contract

## Status

This document defines the explicit mutation boundary between a reviewed `CrossRunReconciliationResult` and the canonical `LayoutDocument`.

Matching/proposal generation and application are separate operations.

The default behavior is non-destructive: producing or reviewing a reconciliation result MUST NOT mutate `LayoutDocument`. Mutation occurs only through an explicit apply operation using an explicit, reviewable decision set.

## 1. Identity authority

`MedicalRegionData.region_id` remains the sole persistent identity of a region in `LayoutDocument`.

When a hypothesis is accepted against an existing persistent region:

- the persistent `region_id` MUST remain unchanged;
- `hypothesis_id` MUST remain analysis-scoped provenance;
- no identifier copying or silent identity replacement is permitted.

A newly approved region receives a new persistent identity according to the document identity policy. It MUST NOT inherit an analyzer hypothesis ID as its persistent identity unless a future explicit contract says otherwise.

## 2. Explicit apply boundary

The system MUST expose a conceptual separation:

```
analysis → matching → reviewed decision set → explicit apply → new LayoutDocument
```

The matcher MUST NOT call the apply operation implicitly.

The apply operation MUST NOT independently invent matches.

An apply request therefore identifies which reconciliation outcomes are explicitly accepted and which remain unresolved.

A future runtime SHOULD use distinct vocabulary such as:

- `ReconciliationApplyDecision`
- `ReconciliationApplyResult`
- `ApplyAction`

These are contract vocabulary, not a requirement to implement the exact names immediately.

## 3. What may be changed

An accepted application against an existing persistent region MAY update only fields explicitly authorized by the apply decision.

At minimum, the decision MUST identify the target persistent `region_id` and the accepted hypothesis.

The implementation MUST NOT infer additional field changes merely because they are present in the hypothesis.

The safe default is:

- geometry/type changes require explicit authorization;
- provenance changes require explicit authorization;
- manual state MUST NOT be cleared implicitly;
- unrelated document regions MUST remain untouched.

Any future field-level update policy must be versioned/documented.

## 4. Manual-region protection

A region marked as manually created or manually modified is protected.

Therefore:

- a `MATCHED` result MUST NOT automatically overwrite a manual region;
- a geometry/type conflict MUST remain `CONFLICT` until explicitly resolved;
- explicit human approval is required before changing protected manual content;
- applying an approved manual change MUST record that the region was explicitly modified.

A matching score, even if unique, MUST NOT override manual protection.

## 5. Outcome application semantics

### MATCHED

A `MATCHED` candidate MAY be applied only when the decision explicitly accepts it and all validation rules pass.

For a non-manual target, the apply policy may authorize specific updates.

For a manual target, `MATCHED` alone is insufficient; explicit protected-region authorization is required.

### UNMATCHED_NEW

This outcome MUST NOT create a persistent region automatically.

Creation requires an explicit apply decision identifying the intended new region and its persistent identity/provenance.

### UNMATCHED_EXISTING

This outcome MUST NOT delete or deactivate the persistent region.

It is review information only unless a separate explicit deletion/deactivation policy is introduced.

### AMBIGUOUS

An ambiguous candidate MUST NOT be applied.

It requires explicit resolution into a single valid action before mutation.

### CONFLICT

A conflict MUST NOT be applied by default.

It requires an explicit resolution that satisfies the applicable protection and identity rules.

## 6. No implicit deletion

The apply layer MUST NOT interpret omission, `UNMATCHED_EXISTING`, analysis failure, or changed analyzer configuration as a deletion command.

Deletion/deactivation, if ever supported, requires a separate explicit action and policy.

This protects manual regions and prevents partial/filtered analysis from becoming destructive.

## 7. Atomicity

An apply operation MUST be atomic from the caller's perspective.

If validation of any selected action fails:

- no partial mutation may be committed;
- the original `LayoutDocument` remains unchanged;
- the failure identifies the invalid decision/action.

The preferred implementation model is copy/validate/commit rather than mutating the canonical document incrementally.

## 8. Idempotency

Applying the same accepted decision set to the same document state SHOULD be idempotent.

A repeated application MUST NOT:

- generate a different persistent region identity;
- duplicate an explicitly created region;
- progressively alter geometry due to repeated application;
- clear or rewrite manual provenance unexpectedly.

If the document has changed since the decision was produced, the apply operation SHOULD fail with a stale/conflict result rather than silently applying against a different state.

## 9. Decision provenance and auditability

Every applied action MUST be traceable to:

- source/document identity;
- analysis run identity;
- hypothesis identity;
- target persistent region identity, when applicable;
- matching policy/configuration identity;
- explicit apply decision;
- applicable validation/protection state.

The system SHOULD preserve enough information to answer:

1. what was proposed;
2. what a human/policy accepted;
3. what was actually applied;
4. which persistent identity was preserved;
5. which fields changed.

Apply provenance is distinct from OCR confidence and cross-run evidence.

## 10. Deterministic application

Given the same document state, reconciliation result, explicit decision set, and apply policy/version, application MUST be deterministic.

The result MUST have stable serialization/order.

The apply layer MUST NOT use hash iteration, incidental list order, UUID randomness, or execution order as an implicit semantic decision.

## 11. Coordinate-space validation

Every geometry change MUST comply with the existing layout coordinate contract:

- source-image pixel space;
- origin top-left;
- +X right;
- +Y down;
- bbox `(x, y, width, height)`;
- nonnegative width/height;
- no implicit DPI conversion;
- no silent clipping or rounding.

The apply layer MUST NOT silently repair invalid geometry.

Whether out-of-image geometry is permitted remains governed by the existing coordinate contract; consumers may explicitly validate/reject/clip according to policy, but application MUST record the chosen policy.

## 12. Serialization and compatibility

Apply decisions/results MUST be explicitly serializable if they are persisted or transported.

Serialization MUST preserve:

- persistent region identity;
- analysis run identity;
- hypothesis identity;
- source/document identity;
- selected action;
- policy/version;
- relevant evidence/provenance.

Existing `LayoutDocument`, `LayoutAnalysisResult`, and cross-run matching payloads MUST remain readable unless an explicit versioned migration is introduced.

## 13. Existing runtime compatibility

The existing `reconcile()` operation remains a non-destructive analysis/reconciliation operation.

Cross-run matching remains a proposal layer.

Apply is a separate layer.

No existing call path should gain mutation merely because an apply-capable runtime is introduced.

## 14. Required runtime safety tests

A future implementation MUST test at least:

1. no mutation occurs when creating/reviewing a reconciliation result;
2. only explicitly accepted actions are applied;
3. persistent `region_id` is preserved;
4. hypothesis IDs never become persistent IDs;
5. manual regions cannot be overwritten without explicit protected-region authorization;
6. `UNMATCHED_NEW` does not auto-create;
7. `UNMATCHED_EXISTING` does not auto-delete;
8. `AMBIGUOUS` and `CONFLICT` cannot be applied without explicit resolution;
9. failed validation leaves the document unchanged;
10. repeated application is idempotent or explicitly rejected as stale;
11. geometry follows the coordinate-space contract;
12. apply provenance survives serialization;
13. unrelated regions remain unchanged;
14. existing legacy serialization remains readable.

## 15. Non-goals

This contract does not define:

- automatic conflict resolution;
- semantic/ML matching;
- OCR execution;
- SQLite/Hugging Face/React integration;
- destructive replace-all reconciliation;
- automatic deletion;
- automatic creation from unmatched hypotheses.

## 16. Architectural principle

The canonical data flow is:

`Analysis → Match Proposal → Review/Policy Decision → Explicit Apply → Validated LayoutDocument`

The apply step is therefore an explicit mutation boundary with identity preservation, manual protection, atomic validation, and auditable provenance.

It is not a hidden continuation of matching.
