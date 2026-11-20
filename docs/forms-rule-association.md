# Form rule association fixes

This candidate changes Forms scores on existing saved outputs. Compare using both the
scorer commit and dataset revision. It extends the released scorer independently; no
colleague implementation was reused. This is an unmerged PR candidate; no release has been published.

## Matching behavior

- Complete flat `question — option` and structured `[question, option]` checkbox groups
  share question ownership. Sparse and ambiguous groups keep contextual matching.
- Printed IDs such as `14a` and `14 a` can match. Unique captions may omit a section
  prefix; conflicting IDs, repeated fields and explicit sections still disambiguate.
- Uniquely owned `caption: value` fields can recover outside GT order. Reordered parent
  sections and explicitly numbered answer blocks retain their question owners.
- Numbered table fields resolve against row and column captions in the same grid,
  including shared Yes/No headers, inherited section IDs and small two-row field grids.
  Scoped two-row grids cross column order only when GT assigns distinct IDs to the
  repeated caption in distinct scopes; lone scoped fields and reused IDs keep context.
  Expected answers do not select the target cell. Duplicate grids remain ambiguous.
- Printed field IDs, cents placeholders and supported inline numbered blank slots stay
  outside entered values. Generic empty HTML cells retain the existing handling.
- A word-only parenthesized caption on the following line completes its exact GT
  caption. Numeric parenthetical fragments keep the existing matching path.
- Date-format header tiers can lead to split date cells. A complete date does not absorb
  a neighbouring N/A cell. Blank answer rows and horizontal blank slots are preserved;
  another column's format hint does not turn the current field into a vertical header.
- Unambiguous Yes/No tick glyphs and LaTeX checkbox marks are recognized. An isolated
  `No √ Yes` remains ambiguous; square roots in mathematical text remain unchanged.
- A currency-marked amount after printed instructions can be read before the next
  caption. Only the first such amount is considered; a later matching amount cannot
  rescue a wrong one.

## Numeric signs and the GT annotation

Signs survive token extraction and printed-frame normalization. An explicit wrong sign
fails in plain text, Markdown tables and HTML tables. Telephone-code parentheses,
identifier hyphens and Markdown list markers retain their formatting roles. Whole
numeric parentheses are accepted as an unsigned printed frame or an accounting negative;
wrong magnitudes and explicit minus signs remain significant.

A text-field rule may opt into `printed_value_prefix: "−"` (ASCII `"-"` is also valid).
Use it only after visually confirming a subtraction operator outside an unsigned answer
box. The scorer can then remove one printed operator; an entered negative or a remaining
second minus still fails against positive GT. Rules without this attribute stay strict.
This attribute does not affect caption ownership or change the expected value.

The local dataset adds this attribute to `l125` in
`forms/filled__canada--01_T1-5006-R--p7__typed.test.json` and synchronizes `forms.jsonl`.
The value remains `4,200.15`. Page 8 line 167 is an entered negative and stays unchanged.
Page 8 line 166 is not an active rule, so this change does not add or broaden one.

## Evidence and reproduction

The replay covers 500 forms and 100 fillable forms, with 33,591 saved outputs across
56 providers (nine outputs are absent). It uses the local GT snapshot plus the single
annotation above. This is a paired scorer replay, not new provider inference or a
leaderboard ranking. Counts, source hashes and test results are recorded in
`forms-rule-association-validation.json`.

Baseline scorer: `a237f0f3684055461b8968d5dc3a4aad04ddd697`.
Saved outputs: `llamaindex/ParseBench-2-results@23ebd7c4fb2a7b61d2ff02e0e955ceee36b863dd`,
`results/<provider>/{forms,fillable_forms}.jsonl`. The additional Qwen thinking provider
uses `21ecc920c6969f69353d4be51d27a4346979f645`. Baseline caches are used only after
scorer and Markdown SHA-256 checks. The replay uses a frozen candidate. A subsequent scope guard was checked by parsing all
33,591 outputs and rescoring every exact two-row/two-column grid; an AST comparison
confirms that only `_rule_table_slots` changed. The validation JSON records that overlay.

The 17 previously flagged decreases were reviewed against the page images and exact
provider output. Fifteen recover; two retain lower scores because the output has wrong
migration ticks or detached ticks without reliable question ownership. This does not
imply every remaining provider error has been fixed or every changed score approved.
A separate 275-pair control checks that swapped field values remain detectable.

The new GT attribute was also replayed separately on all 56 saved outputs of Canada T1
page 7: neither the candidate nor the legacy scorer changed those page scores. Controlled
cases demonstrate the opt-in behavior; no measured provider gain is claimed for the flag.

## Review scope

This is targeted association repair, not a general order-invariant form aligner. Complex
malformed grids and outputs that detach answers from their questions can remain penalized.
The regression suite includes wrong amounts, reversed ticks, swapped rows, conflicting
section IDs, repeated captions, blank dates followed by another person's answer, malformed
signs and ambiguous symbols. The report distinguishes scorer defects from provider errors.

The self-contained source-image report and GT change log are at
`~/Code/explain/parsebench-form-scoring-followup.html`. Earlier audit evidence is retained
in `parsebench-form-sign-and-visual-audit.html` and the archived follow-up report.


## Printed prefixes in blank fields (unreleased)

A sparse text-field rule with `value: ""` may declare a source-verified literal
`printed_value_prefix`, such as `"DE"` in a blank domestic IBAN. The headline Forms
scorer accepts either no entered content or that literal alone. Comparison ignores
whitespace and letter case, so character-box transcription `D E` is accepted.
Additional characters, repeated prefixes and wrong country codes remain errors.
The prefix does not replace a filled answer, alter caption matching or apply to
unannotated fields. The existing subtraction-operator behavior is unchanged.

This metadata is read from `form_field` rules. A page with a FormNode GT tree uses
that tree instead; it must represent printed text as static nodes and writable
slots as fields. Legacy per-rule diagnostics do not apply this headline tolerance.
Dataset v2.07 proposes the blank-IBAN annotation in
[HF PR #8](https://huggingface.co/datasets/llamaindex/ParseBench-2/discussions/8).
CLI 2.1.2 does not yet apply the literal-prefix tolerance. Pin dataset and scorer
versions together; full-provider rescoring for this change has not been run.
