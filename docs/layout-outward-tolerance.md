# Outward layout tolerance

The optional `bbox_tolerance` field on a word-level layout rule relaxes only the tightness component of `layout_grounding_pass_rate`. The existing 0.5–0.9 IoU ramp remains unchanged. Localization, token attribution, grouping, element weights, and extra/duplicate prediction penalties use the original boxes.

```json
"bbox_tolerance": {
  "version": "outward-word-height-10-v1",
  "outward_normalized": [0.001, 0.00075, 0.001, 0.00075]
}
```

The four values are left, top, right and bottom distances in normalized page coordinates. They are outward-only. These example values represent equal physical distances on a 600 × 800 page. Omitted metadata preserves legacy behavior. Unknown versions, negative/nonfinite values and malformed tuples fail validation.

Prepare a local dataset with `python scripts/annotate_layout_outward10.py DATASET_ROOT` (requires PyMuPDF). The script writes sidecars and `layout.jsonl` together, without changing existing boxes or source documents. It computes 10% of physical word height for both axes. Punctuation borrows the height of the nearest alphanumeric word whose vertical extent contains its center and whose horizontal gap is at most three letter heights; otherwise it uses its own height. Margins stop at the page boundary and at half the gap to a neighboring word. Diagonal neighbors are separated on at least one axis. Already-overlapping and rotated words receive zero. This is deterministic geometry, not automatic ink-box repair or manual approval of every annotation.

The scorer transfers the outward growth of a region's member-word hull to the existing region hull. It caps that growth again against nonmember words and the page, since regions may already include font padding. It adjusts each edge only toward the prediction and never contracts the GT. Box-only elements, Pictures, Tables and rotated region/prediction hulls retain legacy tightness. A zero-allowance word in a mixed region contributes no growth, but the region can still benefit from other member words at its boundaries.

A clipped prediction receives no new inward allowance. It can still receive partial/full credit under the existing IoU ramp. The allowance does not repair missing text, incorrect word boxes, bad grouping or missing regions.

## Compatibility

The dataset must opt in; updating the CLI alone does not change old scores. CLI 2.0.4 and earlier ignore this new field and score using legacy tightness. To reproduce scores from an annotated dataset, record both its revision and a scorer revision that implements this contract. This support is available in CLI 2.1.0 and later; CLI 2.0.4 does not apply the allowance.
