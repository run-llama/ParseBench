# Per-point chart tolerance

A mixed chart can print some values while leaving small segments unlabelled. Keep the chart's existing `data_labels_shown` setting and annotate only points that need an explicit estimation margin:

```json
{
  "name": "Private leasing",
  "values": ["2.5", "6"],
  "point_tolerances": [
    {"relative_tolerance": 0.1, "absolute_tolerance": 2},
    null
  ]
}
```

`point_tolerances` belongs to a series and aligns with its `values`. Omit the field, or use `null` at a point, to retain existing scoring. Each override requires both finite, non-negative tolerance numbers. Invalid lengths or entries fail rather than silently changing the annotation.

For an annotated point, the full-credit margin is `max(relative_tolerance * abs(GT), absolute_tolerance)`. The absolute margin uses the chart value's units: for percent values written as 2.5, an absolute margin of 2 means two percentage points. It does not grow with the provider's prediction. Outside that margin, the existing value-score decay applies. Explicit point margins take precedence over the chart default and sparse-rule estimation inference, and do not use the printed-number rounding allowance.

With GT 2.5, relative 0.1 and absolute 2, values 0.5–4.5 receive full numeric credit; 20 does not. Other points retain existing tolerances. Series merged by the colour-group matcher carry their original point margins into the merged representation.

This applies to full chart tables, not sparse `chart_data_point` rules. Older CLI releases ignore the new metadata, so they retain the stricter/default result rather than applying these margins. Datasets using this field need a CLI release containing this change to obtain the intended scores. Existing datasets without the field retain their scoring behaviour.
