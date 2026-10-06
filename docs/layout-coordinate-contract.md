# LayoutDocument Coordinate-Space Contract

**Status:** Proposed contract for review  
**Issue:** #21  
**Applies to:** `LayoutDocument`, `MedicalRegionData`, and consumers of the region-editor model

## 1. Canonical coordinate space

The canonical bounding-box coordinate space is the **source-image pixel coordinate space**.

- Origin: **top-left** of the source image.
- +X: **right**.
- +Y: **down**.
- Units: **source-image pixels**.
- BBox representation: `(x, y, width, height)`.
- BBox values are floating-point numbers so that geometry can be preserved without premature rounding.

For an image with:

```text
image_dimensions = (W, H)
```

the image extent is the rectangle from `(0, 0)` to `(W, H)` in this coordinate space.

## 2. BBox semantics

For a region:

```text
bbox = (x, y, width, height)
```

the rectangle starts at `(x, y)` and extends by `width` horizontally and `height` vertically.

- `width >= 0`
- `height >= 0`
- No implicit conversion to `x1, y1, x2, y2`.
- No implicit normalization or rounding during serialization.

Interactive UI may impose a practical minimum size for usability. That UI constraint is **not** the canonical model constraint.

## 3. Image dimensions and DPI

`image_dimensions` describes the source image in pixels:

```text
(width, height)
```

`dpi` is metadata. It does **not** change the coordinate system of a bbox.

A consumer that needs physical units, PDF points, normalized coordinates, display pixels, or another coordinate system must perform an explicit conversion.

## 4. Coordinate conversions

Conversions between coordinate systems MUST be explicit.

Examples include:

- normalized `[0,1]` coordinates
- PDF points
- document points
- Qt/device/display pixels
- resized-image coordinates
- OCR-engine-specific coordinate systems

A conversion must document both the source and destination coordinate spaces and must not silently mutate the stored canonical bbox.

## 5. Out-of-image regions

The core model does **not** silently clip a region to the image boundary.

A region may therefore be partially outside the image coordinate extent when produced by an upstream inference/import workflow.

Consumers may reject, clip for presentation, or otherwise handle such a region, but that operation must be explicit and must not overwrite the canonical bbox without an intentional edit.

This keeps inference/import provenance intact and avoids destructive geometry changes.

## 6. UI synchronization

The PySide6 region editor operates in the same source-image pixel coordinate space as the model.

The editor MUST preserve:

```text
model bbox <-> scene geometry
```

without an implicit coordinate-system conversion.

Display scaling/zoom affects presentation only.

High-DPI rendering must not be interpreted as a change in the document coordinate system.

## 7. Serialization

The serialized `bbox` remains:

```json
[ x, y, width, height ]
```

and represents the canonical source-image pixel coordinates.

Serialization/deserialization MUST NOT:

- round coordinates,
- clip to image dimensions,
- convert to normalized coordinates,
- infer a different coordinate space from DPI.

## 8. Versioning

`layout_version` identifies the layout/serialization contract.

It is independent of:

- OCR engine versions
- confidence calibration versions
- UI versions
- model/training dataset versions

Any incompatible change to the coordinate or serialization contract requires an explicit layout-version decision.

## 9. Confidence is orthogonal

BBox geometry and OCR confidence provenance are separate concerns.

A region's `confidence_evidence` MUST NOT be interpreted as evidence about geometric accuracy unless a future contract explicitly introduces such evidence.

## 10. Non-goals

This contract does not define:

- layout inference algorithms,
- OCR engine selection,
- OCR execution,
- training database schemas,
- Hugging Face export formats,
- React/web editor behavior.

Those layers may consume this contract but must not redefine the canonical coordinate space independently.

## 11. Implementation policy

This document establishes the contract first.

Behavioral enforcement or additional validation should be introduced only with:

1. explicit tests,
2. compatibility analysis against existing `LayoutDocument` data,
3. evidence that the change is non-destructive,
4. a separate reviewable change when it affects runtime behavior.

