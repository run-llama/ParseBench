# Forms scorer corrections for the v2.03 GT candidate

Base: upstream main `4cb2d7461ee34bce4aa6b9de5b5e2d820296d5ab`. PR #5405 is excluded. This patch changes scoring association, not GT answers, loss weights, blank-omission policy or published results.

Implemented behavior:

- Repeated printed IDs can connect an instruction-bearing caption to a later explicit ID-only value line. Matching is bounded to local fields and does not search for the expected answer.
- Two-row Markdown tables without outer pipes are recognized. Values above a caption-only footer row are read by column, including partial references whose neighbouring captions are explicitly parenthesized.
- A row description containing a column name does not become a repeated-header block merely by containing those words.
- A parenthesized suffix that completes the exact GT caption remains part of the caption, including printed codes such as `(D1)`.
- Repeated checkbox options follow their recorded question context even when option order differs. Prose mentions are excluded; overlapping caption alternatives are deduplicated. Unresolved ownership collisions defer repair rather than introduce a new omission.
- Partial native trees can declare `metadata.forms.reference_scope = "mapped_fields_only"`. This disables extra-field detection only; the annotated fields still receive normal scoring. Absent this declaration, native trees retain exhaustive-reference behavior. The frozen GT candidate used for the replay did not add this opt-in.

Validation: 454 forms tests pass (68 new), Ruff checks pass, and both parsers' saved 500-page forms outputs were replayed on identical candidate GT. Agentic Plus: 83.9339 → 84.2448, 8 pages improve and none decline. Reducto: 81.9407 → 82.2867, 10 improve and none decline. These are forms-subset averages, not published overall benchmark scores. Score gains do not establish correctness; negative controls check wrong values, ticks, IDs, missing values and row swaps.

```sh
python -m pytest -q tests/parse_bench/evaluation/metrics/parse/test_forms_v2*.py
ruff check src/parse_bench/evaluation/metrics/parse/forms_v2.py tests/parse_bench/evaluation/metrics/parse/test_forms_v203*.py
```

The evidence package includes the exact frozen scorer used for replay, the current source, source/GT provenance, saved Markdown, replay scripts, before/after events, controls, validation and tests. The current scorer differs from the replay snapshot only by an explicit default binding of the footer helper's loop variable to satisfy lint; the footer tests were rerun afterward.

Remaining limitations: absent or ambiguous question contexts, arbitrarily transposed layouts, multilingual instruction variants beyond the bounded repeated-ID helper, and detached printed item numbers. The isolated E2 control with only one selected option is not repaired; complete contextual option groups are tested. The broader GT candidate still requires review of automated mappings. No score policy, publication or inference rerun is part of this patch.


## Confirmed blank-field policy

For a blank text field, preserve its printed label. An empty-value marker is optional:

```text
Name: Pierre
Middle name: [empty]
```

and

```text
Name: Pierre
Middle name:
```

are both accepted. Whitespace-only values and `[___]` are accepted too. Omitting the `Middle name` label remains a missing-field error. An invented value remains an error even when preceded by an empty marker. A following field's value must not be read into the blank slot.

This confirms existing behavior; it does not waive missing labels or change omission weights. Regression tests cover plain text, bold labels and table layouts, with both sparse-rule and native-tree references. It concerns blank text fields; checkbox and signature states retain their separate rules.
