# Forms scorer corrections for reviewed v2.03 GT

Base: upstream main `4cb2d7461ee34bce4aa6b9de5b5e2d820296d5ab`. Experimental PR #5405 is excluded. This patch changes field association and bounded output handling; it does not change expected answers, loss weights, or blank-omission policy.

## Behavior

- Repeated checkbox options use question/group structure already present in GT. Unique full questions override shared instruction suffixes when output order differs. Complete, uniquely located option groups move atomically; competing occurrences or externally owned tokens do not become false omissions.
- Plain text and lists do not need to reproduce source geometry. Tables additionally preserve same-cell, column and row ownership. Separate short option and marker cells can establish a consistent alternating direction. Mixed inline/table cells, composite captions, intervening text/marks and incomplete pairs do not establish that direction.
- Local options containing nested text fields are preserved rather than replaced with a nearby cleaner-looking Yes/No occurrence. Expected tick states never select the association.
- Repeated printed IDs can connect an instruction-bearing caption to its local value line or table cell. Explicit `16: 6,413` and, after instructions, `16 6,413` are supported. Wrong IDs/values remain errors. A grouping comma in `$12,400` is not an ID/value delimiter. Empty value cells are not skipped to reach instructions or another field. Shared row IDs and extra answer columns decline the new table repair. Three glyphs around two options do not establish marker direction.
- Two-row pipe footers with explicit parenthesized captions can place their values above their captions. Ordinary stacked blank form labels and declaration prose do not qualify.
- Row descriptions containing column-name words do not become repeated headers. Parenthesized suffixes completing an exact GT caption stay part of that caption, including `(D1)`.
- Partial native trees may opt into `metadata.forms.reference_scope = "mapped_fields_only"`, which disables only extra-field detection. All annotated fields still receive normal scoring. The reviewed dataset does not add this opt-in.
- Exact caption-match results are reused for identical token windows in repeated hallucinated prose. Field-line regexes are skipped when the required colon is absent. These avoid pathological runtime without truncating outputs or changing comparisons.

## Blank-field policy

The printed label is required; an empty-value marker is optional. `Middle name: [empty]`, `Middle name:`, whitespace-only values and `[___]` are accepted for a blank text field. Omitting the label remains a missing-field error. Invented values remain errors, including values after empty markers. Checkboxes and signatures retain their own state rules.

## Validation

- 496 forms tests pass; Ruff and diff whitespace checks pass. Fifteen GT builder/review tests pass.
- 32,993 saved outputs from all 55 published parser pipelines were rescored: 500 forms + 100 fillable cases per pipeline where published. Seven outputs are absent from the published results; 242 present empty outputs are scored normally. No final evaluation errors.
- All 500 forms and 100 fillable canonical reference renders pass. All retained answers/types, 600 PDFs and 100 fillable sidecars are unchanged.
- 1 scorer-only decrease remains after review: LandingAI's India PAN page explicitly reverses question 12's checked Yes/No states. The source PDF confirms No is checked. This decrease is retained as a real parser error.
- Across parsers, scorer-only forms-dimension changes range from +0.0119 to +0.3797 points. Per-parser/subset results, coverage gaps and exact events are in the attached audit summary. GT-only decreases (3,950 output cases on 314 pages) are recorded separately, not waived.
- Final scorer SHA256: `722f7440eff21ace1aacf662dbcf09c3d24c02a1e4f93e6d1e7bbb852e63e20c`. Candidate manifest SHA256: `8fc7857365a09392606f8f14c8a5caf8c231bdca8d498da01ed3885871c322af`.
- The public baseline had one timeout on pathological repeated prose; it was replayed with the identical public scorer and a longer per-arm limit. The original timeout record is retained. Candidate07 changed one day-specific label: all 55 affected baseline outputs were recomputed, and the other 599 GT sidecars were verified byte-identical. The final scorer arm was rerun in full on candidate07.

The replay compares three arms on identical saved outputs: original GT/public scorer; reviewed GT/public scorer; reviewed GT/final scorer. The forms dimension uses 7/8 forms + 1/8 fillable forms. Other dimensions were not rerun, and no overall published leaderboard score is claimed. No parser inference was rerun.

Source dataset: `llamaindex/ParseBench-2` at `e7db5227a2f0ec6e29ba7808631cf3925e0ed678`. Reviewed GT: HF PR #4 at `5a1a14f739db073dda34779d0d38de363e401ca4` (verified candidate07 data; subsequent commits add audit artifacts only). Saved outputs: `llamaindex/ParseBench-2-results` at `23ebd7c4fb2a7b61d2ff02e0e955ceee36b863dd`. Exact source/input hashes, coverage gaps and per-parser results are included in the audit summary.

```sh
python -m pytest -q tests/parse_bench/evaluation/metrics/parse/test_forms*.py
ruff check src/parse_bench/evaluation/metrics/parse/forms_v2.py tests/parse_bench/evaluation/metrics/parse/test_forms_v203*.py
```

## Remaining limits

A lone tick between No and Yes does not establish direction by itself. Detached answer columns without row links, missing/ambiguous question captions, competing complete pairs, incomplete groups and arbitrary transpositions retain the existing scoring path. This does not certify those existing scores as correct. GT grouping cannot reconstruct information absent from the parser output, and expected answers must not be used to guess an association.

GT-only decreases are recorded separately from scorer-only decreases. The prior annotation review covered proposed transfers and a 109-page two-parser regression ledger; 28 additional worst-parser large-drop page cases outside that ledger were reviewed. One Brazilian day-only deadline label was restored because the shared date caption lost day/month/year identity. This review does not manually certify every unchanged annotation or every newly exposed legacy scoring limitation. Scorer release: `2.0.4`. The dataset PR remains separate and must be merged independently.
