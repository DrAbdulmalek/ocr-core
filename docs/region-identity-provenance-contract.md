# Region Identity and Provenance Contract

## Status

This document defines the contract for persistent region identity, analysis hypothesis identity, analysis-run identity, and provenance. It is documentation-first for Issue #29 and does not change runtime or serialization by itself.

## 1. Three distinct identities

### 1.1 Persistent region identity

MedicalRegionData.region_id is the identity of a persisted region instance inside a LayoutDocument.

A persistent region ID:
- identifies the same persisted region across save/load;
- survives non-destructive analysis reruns;
- survives manual edits;
- MUST NOT be replaced merely because an analyzer emits a different hypothesis ID;
- is owned by the persisted document model, not by an analyzer.

The existing region_id field remains the compatibility anchor. Any future rename or schema change requires explicit versioning/migration.

### 1.2 Analysis hypothesis identity

An analysis hypothesis identifies one candidate region produced by one analysis result.

For the current runtime, RegionHypothesis.region_id is treated semantically as an analysis-scoped hypothesis identifier, not as proof of persistent semantic identity.

A hypothesis ID:
- MUST be unique within one LayoutAnalysisResult;
- MAY be deterministic for a given analyzer/configuration/source;
- MUST NOT be interpreted as persistent document-region identity;
- MUST NOT establish cross-run identity merely because the string matches;
- may change when configuration, input evidence, ordering, or algorithm changes.

Future runtime work MAY rename this field to hypothesis_id, but that is an API/compatibility change outside this documentation-only change.

### 1.3 Analysis-run identity

An analysis run is one execution event producing one LayoutAnalysisResult.

A future runtime representation SHOULD provide an explicit analysis_run_id that uniquely identifies that event.

The analysis run is distinct from analyzer_version, configuration_id, source_id, and layout_version. Those fields describe the run; they do not replace its event identity.

## 2. Cross-run matching

Cross-run matching is an explicit reconciliation operation.

The system MUST NOT infer persistent identity from:
- matching hypothesis IDs alone;
- list position;
- incidental geometry equality;
- region type equality;
- analyzer execution order.

A hypothesis MAY be mapped to an existing persistent region only through an explicit matching/reconciliation decision with documented evidence.

If no safe match exists, the result remains a proposal for a new persistent region rather than silently replacing an existing region.

## 3. Human edits have precedence

A manually created or manually modified persistent region MUST survive a later analysis rerun by default.

If analysis proposes conflicting geometry/type for a manually edited region, reconciliation MUST represent the conflict or update proposal without mutating the canonical document.

Absence of a manual region from a later analysis MUST NOT authorize deletion.

## 4. Provenance minimum

The provenance model MUST distinguish at least:
- inferred/analyzer-created;
- imported/external;
- manually created;
- manually modified.

For analyzer-derived hypotheses, provenance MUST identify:
- analyzer identity;
- analyzer version;
- configuration identity/version;
- source identity;
- analysis-run identity once available.

For persisted regions, provenance MUST survive serialization and MUST NOT be silently reinterpreted during reruns.

The compatibility fields source and is_manually_edited remain valid compatibility primitives. Future richer provenance MUST NOT silently change their historical meaning.

## 5. Determinism and identity

Deterministic analyzer output is desirable for reproducibility, but determinism is not semantic identity.

Therefore:
- deterministic hypothesis IDs are valid within the defined analysis scope;
- a deterministic ID MUST NOT be promoted automatically to persistent region ID;
- configuration changes MAY produce different hypothesis IDs;
- reproducibility is evaluated using complete analysis input/provenance/configuration.

## 6. Serialization and versioning

The canonical persisted layout remains LayoutDocument.

Identity/provenance additions MUST:
1. preserve existing documents unless explicit migration is required;
2. be versioned when serialization shape changes;
3. round-trip without loss of persistent region identity;
4. preserve manual-edit state and provenance;
5. avoid silently converting hypothesis identity into persistent identity.

layout_version describes the layout document schema/contract. It is independent from OCR confidence calibration and analyzer configuration/version.

## 7. Reconciliation safety rules

Reconciliation MUST remain non-destructive by default.

Minimum safe actions:
- unchanged;
- add proposal;
- update proposal;
- conflict;
- deletion proposal.

Destructive replacement, if ever introduced, MUST be explicit higher-level policy and MUST NOT be an implicit side effect of rerunning analysis.

## 8. Compatibility with current runtime

The current implementation is compatible with this contract:
- LayoutDocument remains the canonical persisted object;
- MedicalRegionData.region_id remains persistent document identity;
- RegionHypothesis.region_id is analysis-scoped identity;
- reconciliation matches only persisted IDs and does not infer geometry/list-position identity;
- manually edited regions are protected;
- analysis results are immutable and deterministic;
- reconciliation does not mutate the canonical document.

A future runtime change is required to introduce explicit analysis_run_id and, if desired, a dedicated hypothesis_id field. That change MUST be separate and include compatibility tests.

## 9. Required tests for runtime implementation

Before production Medical Layout Analyzer work, runtime tests SHOULD prove:
- repeated identical runs are reproducible;
- different runs cannot silently replace persistent region IDs;
- manual regions survive omitted analyzer hypotheses;
- manual edits produce conflicts rather than silent overwrite;
- provenance survives serialization;
- hypothesis-to-region mapping is explicit;
- analyzer/configuration changes do not silently mutate canonical identity;
- legacy documents remain readable or have an explicit migration path.

## 10. Non-goals

This contract does not define:
- a specific ML/layout model;
- OCR engine migration;
- SQLite or training-database schema;
- Hugging Face export;
- React editor implementation;
- automatic semantic region matching;
- destructive document replacement.

Those require separate design and review.