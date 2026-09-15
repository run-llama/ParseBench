# text_extended v2.1 scoring contract

Rules opt in with `text_normalization: "text-v2.1"`. Rules without this field
retain legacy behavior. Do not compare historical and revised scores as a model
improvement: the content being measured and its tokenization have changed.

`text_v21.reference_rules()` is the single authoring and evaluation contract.
It reads the reference Markdown, never a parser prediction. To regenerate:

```
python scripts/regenerate_text_v21.py DATASET --write
python scripts/regenerate_text_v21.py DATASET
```

The second command verifies freshness and reference self-consistency without
writing. Always force reevaluation after changing annotations or evaluators.
Self-consistency is necessary; it does not prove that an annotation matches its PDF.

## Content and units

- Markdown labels are text; link targets, image alt descriptions, and image URLs
  are not. Ordinary code bodies and table cells are included. Generated `mermaid`
  and `description` fences are excluded. Language labels and markup are excluded.
- Structured page headers and footers join the corresponding page body for these
  content rules. An identical boundary copy is counted once; body repetitions are
  retained. Header/footer classification and other structural rules remain separate.
- Word bags use NFC and lowercase. Single letters and digits are retained, so answer
  letters count. Underscores delimit identifiers. CJK is measured in characters;
  Latin/Indic runs remain tokens with combining marks preserved. This measures
  content coverage, not CJK word segmentation. Source script substitutions count as errors.
- Sentence bags contain ordered token spans, including short answer-number pairs.
  Whitespace and punctuation are not required. Counts are measured against the full
  reference too, so repeated or nested anchors cannot fail on identical input.
  Sentence metrics are span coverage/order signals, not linguistic sentence accuracy.
- Digit bags count ASCII digit occurrences in the same projection. Their score is
  `1 - (missing + excess)/(expected + observed)`; numeric values and their semantic
  associations need the word/span rules as well.
- Recall counts required occurrences. Precision penalizes unreferenced content;
  repeated expected text is measured by the separate duplicate metric. Empty output
  has zero recall; precision without observed content is vacuously one.

These metrics do not establish formula equivalence, exact punctuation, table
structure, image correctness, reading order across unrelated spans, or document
layout. Keep the corresponding dedicated rules and source review.

