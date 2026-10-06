# Layout Analysis and Region Lifecycle Contract

**Status:** Proposed contract for review  
**Issue:** #23  
**Applies to:** future layout-analysis producers and consumers of `LayoutDocument`

## 1. Boundary

A layout analyzer consumes source-image evidence and produces layout hypotheses as region geometry and metadata.

The analyzer MUST NOT become the source of truth for persisted training data, OCR text, or human review state.

The canonical persisted layout object remains `LayoutDocument`.

Conceptually:

```text
source image / upstream evidence
        |
        v
   Layout Analyzer
        |
        v
region hypotheses
        |
        v
   LayoutDocument
        |
        +--> human editor
        +--> OCR / downstream consumers
```

An analyzer may produce a new document or an explicit analysis result, but it must not silently overwrite an existing human-edited document.

## 2. Coordinate-space compliance

All analyzer bboxes MUST use the coordinate contract defined by `docs/layout-coordinate-contract.md`:

- source-image pixel coordinates,
- origin at top-left,
- +X right,
- +Y down,
- `(x, y, width, height)`,
- floating-point geometry,
- no implicit clipping, normalization, rounding, or DPI conversion.

An engine-specific coordinate system MUST be converted explicitly before its geometry enters `LayoutDocument`.

## 3. Region identity

A region identity is a stable identifier for one persisted region instance.

Analyzer output MUST NOT assume that a newly generated UUID is equivalent to semantic identity across independent analysis runs.

If cross-run matching is required, it must be an explicit matching operation based on documented evidence such as geometry, type, ordering, or upstream identifiers.

A rerun MUST NOT silently replace manually edited regions merely because a new analysis produced different IDs.

## 4. Provenance

Every inferred region must remain distinguishable from a manually created or manually edited region.

At minimum, downstream implementations need to preserve the distinction between:

- analyzer/inference origin,
- imported/external origin,
- manual creation,
- manual modification.

Existing `source` and `is_manually_edited` fields are useful compatibility primitives, but a future richer provenance model should not reinterpret them silently.

Provenance MUST survive serialization.

## 5. Human-edit protection

Manual edits have precedence over automatic reruns unless an explicit user-approved reconciliation operation is requested.

A normal analyzer invocation MUST NOT:

- overwrite `text_verbatim`,
- overwrite `normalized_view`,
- discard manual bbox edits,
- reset region type changes,
- erase provenance,
- silently delete human-created regions.

If automatic reconciliation is introduced, it must produce an explicit diff/reviewable result rather than mutating the canonical document implicitly.

## 6. Region ordering

`LayoutDocument.regions` is an ordered collection.

The current model does not define a semantic reading-order algorithm. Therefore, consumers MUST NOT assume that list order is a medically meaningful reading order unless an explicit contract establishes it.

A future reading-order field or contract must distinguish:

- storage order,
- geometric order,
- inferred reading order.

Sorting regions for display MUST NOT silently rewrite canonical ordering.

## 7. Geometry relationships

The core contract does not require regions to be:

- disjoint,
- non-overlapping,
- rectangularly non-nested,
- fully inside the image.

Overlap, containment, touching edges, and out-of-image geometry may be legitimate intermediate inference states.

A consumer may reject or filter geometry for a specific task, but such filtering must be explicit and non-destructive.

## 8. Degenerate detections

Analyzer output should reject invalid geometry before it becomes a persisted region.

At the model boundary:

- width and height MUST be non-negative;
- malformed bbox arity is invalid;
- non-finite numeric values MUST NOT be persisted;
- image dimensions remain positive.

A zero-area region is not automatically equivalent to a valid annotation. Future analyzer implementations should define whether zero-area detections are rejected at the analyzer boundary or retained as diagnostic evidence; they must not be silently treated as normal regions.

## 9. Confidence separation

Layout confidence and OCR/text confidence are different evidence domains.

`ConfidenceEvidence` attached to a region MUST NOT be interpreted as:

- OCR text confidence,
- geometric accuracy,
- clinical correctness,
- cross-engine comparable probability.

If a future analyzer emits geometry confidence, it needs an explicit provenance-bearing representation rather than overloading OCR confidence fields.

## 10. Determinism and reproducibility

For a fixed:

- source image,
- analyzer version,
- configuration,
- model/version,
- preprocessing configuration,

the analyzer should produce reproducible output or explicitly document the source of nondeterminism.

Reproducibility metadata belongs to analysis provenance, not to the bbox coordinate system.

An implementation MUST NOT hide nondeterministic matching behind stable-looking region IDs.

## 11. Rerun semantics

A rerun of layout analysis is a new analysis event.

It MUST NOT be equivalent to:

```text
analyze -> replace all existing regions
```

unless the caller explicitly requests destructive replacement and the operation is guarded by a separate policy.

Preferred future behavior:

```text
existing LayoutDocument
        +
new analysis result
        |
        v
explicit reconciliation
        |
        +--> unchanged
        +--> added
        +--> geometry/type proposal
        +--> conflict requiring review
        +--> explicit deletion proposal
```

Human-edited regions should default to conflict/protection rather than automatic replacement.

## 12. Serialization and versioning

Analyzer-produced regions must serialize through the existing `LayoutDocument` contract.

The analyzer must not introduce a competing bbox schema.

Any incompatible change to region serialization or coordinate semantics requires an explicit `layout_version` decision.

Analyzer/model versions and layout versions remain separate dimensions.

## 13. Non-goals

This contract does not define:

- a specific layout model,
- YOLO/Detectron/Transformer/Paddle/other engine selection,
- OCR execution,
- text correction,
- SQLite,
- Hugging Face export,
- React/web editing,
- medical diagnosis,
- clinical interpretation.

## 14. Implementation gate

This document is a contract only.

A runtime analyzer or reconciliation layer must be introduced separately with:

1. unit and integration tests,
2. compatibility tests against existing `LayoutDocument` serialization,
3. explicit provenance tests,
4. rerun/manual-edit protection tests,
5. deterministic/reproducibility evidence where applicable,
6. non-destructive evidence,
7. a reviewable PR separate from this documentation change.
