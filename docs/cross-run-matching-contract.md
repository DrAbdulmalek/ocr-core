# Cross-Run Matching and Non-Destructive Reconciliation Contract

## Status

This document defines how hypotheses from one layout-analysis run MAY be matched to persistent regions from another run.

It extends the Region Identity and Provenance Contract. It is contract-first: it defines evidence, candidate matches, ambiguity, one-to-one constraints, and reconciliation outcomes. It does not implement automatic semantic matching and does not mutate `LayoutDocument`.

## 1. Identity boundaries

The following identities remain distinct:

| Identity | Scope | Authority |
|---|---|---|
| `MedicalRegionData.region_id` | persisted `LayoutDocument` | canonical persistent region identity |
| `RegionHypothesis.region_id` / `hypothesis_id` | one `LayoutAnalysisResult` | analysis-scoped hypothesis identity |
| `analysis_run_id` | one analysis execution event | analysis-run identity |
| `layout_version` | persisted document schema/contract | serialization/layout compatibility |
| analyzer/version/configuration/source | descriptive run provenance | evidence about how a hypothesis was produced |

No equality between two of these identifiers establishes semantic identity by itself.

## 2. Explicit matching model

Cross-run matching MUST be represented as an explicit proposal rather than inferred from identifier equality.

Conceptually, a match proposal contains:

- the analysis run;
- the hypothesis identity;
- the target persistent `region_id`, when one is proposed;
- the matching evidence/reasons;
- a deterministic score or confidence representation, if scoring is used;
- the decision state;
- enough provenance to reproduce or audit the proposal.

The proposal is not itself a mutation of `LayoutDocument`.

A future runtime model SHOULD use distinct names such as:

- `HypothesisMatchCandidate`
- `MatchEvidence`
- `MatchDecision`
- `CrossRunReconciliationResult`

These names are illustrative contract vocabulary; this document does not require their immediate implementation.

## 3. Allowed matching evidence

A future matcher MAY consider multiple explicit evidence dimensions.

### 3.1 Document/source identity

The strongest contextual boundary is the identity of the source document/image or an explicitly versioned source.

A matcher MUST NOT match regions across unrelated documents merely because their geometry or type looks similar.

If source identity is unavailable or ambiguous, the match MUST be treated as lower-confidence or unresolved according to explicit policy.

### 3.2 Geometry

Geometry MAY contribute evidence when both sides use the same coordinate-space contract.

The canonical coordinate space is the source-image pixel space defined by the existing layout contract.

Geometry comparison MUST:
- use explicit tolerances;
- preserve floating-point values without silent rounding;
- never silently convert DPI/display/PDF coordinates;
- record the tolerance or matching configuration as provenance.

Geometry similarity is evidence, not identity.

### 3.3 Region type

Region type MAY contribute supporting evidence.

Equal `RegionType` MUST NOT be sufficient to establish identity.

A type change MAY be a legitimate update proposal or conflict and MUST be represented explicitly.

### 3.4 Stable external anchors

A future application MAY supply a stable external anchor when one exists, such as a document-level annotation identifier.

Such anchors MUST have their own documented ownership and lifecycle.

An analyzer-generated hypothesis ID is NOT a stable external anchor.

### 3.5 Analyzer/configuration provenance

Analyzer identity, version, configuration, and run identity MAY be used to explain or reproduce a proposal.

They MUST NOT be treated as semantic region identity.

Changing analyzer/configuration MUST NOT silently rewrite persistent region IDs.

## 4. Evidence composition

A matcher MUST make its evidence policy explicit.

It MUST NOT use an undocumented weighted formula whose meaning changes between versions.

If a numeric score is introduced:
- its scale and interpretation MUST be documented;
- the configuration/version producing it MUST be recorded;
- thresholds MUST be explicit;
- scores from different matching algorithms MUST NOT be assumed comparable without a documented calibration contract.

Where evidence is insufficient, the safe result is unresolved/ambiguous rather than an automatic match.

## 5. Candidate states

At minimum, a cross-run matching result MUST distinguish:

### MATCHED
One hypothesis has one explicitly accepted persistent-region target.

### UNMATCHED_NEW
A hypothesis has no accepted existing-region target and may become a proposal for a new persistent region.

### UNMATCHED_EXISTING
A persisted region has no accepted hypothesis in the new run.

This is NOT automatically a deletion decision.

### AMBIGUOUS
A hypothesis has multiple plausible persistent-region candidates, or a persistent region has multiple plausible hypotheses, without a unique explicit decision.

### CONFLICT
The candidate evidence or lifecycle state conflicts with a protected/manual region or another explicit invariant.

These states describe reconciliation results; they do not directly mutate the canonical document.

## 6. One-to-one matching invariant

Within one reconciliation operation, the default mapping MUST be one-to-one:

- one hypothesis MUST NOT be accepted for multiple persistent regions;
- one persistent region MUST NOT be accepted for multiple hypotheses.

If several candidates compete for the same target, the result MUST preserve the competing proposals and mark the situation ambiguous/conflicting until an explicit deterministic policy or human decision resolves it.

Any future many-to-one or one-to-many mapping requires a separate documented semantic model. It MUST NOT emerge accidentally from the matcher.

## 7. Manual-region protection

A persistent region marked as manually created or manually modified has precedence over analyzer inference.

Therefore:

- a new hypothesis MUST NOT silently replace its `region_id`;
- conflicting geometry/type MUST produce an explicit update proposal or conflict;
- an omitted manual region MUST remain in the canonical document;
- matching evidence MUST NOT override manual protection merely because its score is higher.

A human-approved decision MAY later authorize an explicit mutation through a higher-level workflow.

## 8. Deterministic candidate generation

For identical inputs, provenance, configuration, and matching policy, candidate generation MUST be deterministic.

Determinism includes:
- stable ordering of candidates;
- stable serialization;
- stable tie representation;
- explicit tie-breaking rules when policy permits automatic selection.

If a tie cannot be resolved safely, it MUST remain `AMBIGUOUS`.

Execution order, dictionary/hash iteration order, or incidental UUID generation MUST NOT decide a semantic match.

## 9. Reconciliation output

Cross-run reconciliation SHOULD produce an immutable result containing enough information to audit every decision.

The result SHOULD distinguish at least:

- accepted matches;
- unresolved/ambiguous candidates;
- conflicts;
- new-region proposals;
- existing regions with no matching hypothesis;
- provenance of the analysis run and matching configuration.

The result MUST NOT modify `LayoutDocument` as a side effect.

Applying accepted changes is a separate explicit operation with its own validation and tests.

## 10. No implicit deletion

An absent hypothesis MUST NOT imply deletion of a persistent region.

Deletion requires an explicit higher-level policy.

This is especially important for:
- manually created regions;
- manually modified regions;
- temporary analyzer failures;
- partial/filtered analysis runs;
- changes in analyzer configuration.

An `UNMATCHED_EXISTING` result is evidence for review, not a deletion command.

## 11. No implicit persistent-ID replacement

When a new hypothesis is matched to an existing persistent region, the persistent `region_id` remains authoritative.

The hypothesis ID remains provenance for the analysis result.

The matcher MUST NOT copy a hypothesis ID into `MedicalRegionData.region_id`.

If a new persistent region is explicitly approved, it receives a new persistent identity according to the document-level identity policy.

## 12. Coordinate-space compliance

All geometry evidence MUST comply with the existing source-image pixel coordinate contract:

- origin at top-left;
- +X to the right;
- +Y downward;
- bbox = `(x, y, width, height)`;
- width/height nonnegative;
- no implicit DPI conversion;
- no silent clipping or rounding.

A conversion from another coordinate space MUST be explicit and versioned before geometry participates in matching.

## 13. Serialization and reproducibility

Any persisted or transport representation of matching proposals MUST preserve:

- source/document identity;
- analysis-run identity;
- hypothesis identity;
- target persistent region identity, when present;
- matching policy/configuration identity;
- evidence and decision state.

Adding these fields MUST preserve readability of existing `LayoutAnalysisResult` and `LayoutDocument` data unless an explicit versioned migration is introduced.

## 14. Separation from OCR confidence

Cross-run matching confidence is not OCR confidence.

The system MUST NOT reuse `ConfidenceEvidence` to represent region-match confidence unless a separate, explicit semantic field is introduced.

OCR engine confidence describes OCR evidence. Match evidence describes identity/reconciliation evidence. They have different scopes and calibration requirements.

## 15. Safe future runtime gate

A runtime implementation may proceed only after tests cover:

1. identical runs produce deterministic candidate ordering;
2. matching never relies on hypothesis-ID equality;
3. one-to-one constraints are enforced;
4. ambiguous candidates remain unresolved;
5. manual regions survive omitted/conflicting hypotheses;
6. hypothesis IDs never replace persistent region IDs;
7. unmatched existing regions are not deleted;
8. geometry matching respects coordinate-space rules;
9. provenance and matching configuration survive serialization;
10. canonical `LayoutDocument` remains unchanged until an explicit apply step.

## 16. Non-goals

This contract does not define:

- a specific ML or heuristic matcher;
- a mandatory numeric similarity formula;
- automatic approval of matches;
- destructive replace-all reconciliation;
- OCR engine migration;
- SQLite/Hugging Face/React integration.

Those require separate implementation and review.

## 17. Architectural principle

The safe direction is:

`Analysis Run A → hypotheses → explicit cross-run matching proposal ← hypotheses ← Analysis Run B`

followed by:

`matching result → human/policy decision → explicit apply → LayoutDocument`

The matching layer is therefore an auditable proposal layer, not a hidden identity-rewrite mechanism.
