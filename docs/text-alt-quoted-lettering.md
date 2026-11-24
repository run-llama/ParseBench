# Quoted lettering in image descriptions

A provider can return `![photo: poster reading "printed words" (citation) above a hilltop silhouette](image.jpg)`.
When the alt contains enough reference text to recover it as transcription, the scorer previously charged the surrounding description as inserted text too. An unrelated reference correction could change the alignment and turn that description into an invented-passage event.

For one explicit quotation introduced by reading, reads, or says, recover the complete quotation and immediately adjacent parenthetical citation. The quotation must meet the existing minimum length and reference-overlap thresholds. Preserve errors within that span. Multiple quotations, unquoted alts, short alts, junk, and unmarked body text retain their existing handling.

## Validation

Issue #316, GT316-027: published `llamaparse_agentic_plus`, `text/text_dense__livingWord`, with corrected v2.09 reference: 83.4227 before, 95.0000 after. The remaining event is block order. With the old erroneous reference, the fixed scorer gives 93.4240. These are page scores, not aggregate leaderboard changes.

The regression test fails on main (83.33 instead of 100 on the reduced correct-transcription example). It passes with the fix and checks that a changed word remains penalized. All 321 text-next tests pass. Full parse-metric collection requires optional table dependencies absent in this environment. A full published-output replay has not been performed.
