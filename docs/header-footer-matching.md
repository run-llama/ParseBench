# Header/footer matching contracts

Header/footer rules may set `section_matching: "nested-v1"`. Without the field (or with `"legacy"`), raw-Markdown matching uses the original behavior. This keeps old rules reproducible with the updated scorer.

The new matcher extracts one complete `<page_header>` or `<page_footer>` block and searches within it. Nested formatting and page-number tags elsewhere in that block no longer cause false failures. Matching cannot cross a header/footer or sub-page boundary. The query text, whitespace and dash normalization are unchanged. Structured page sections remain the preferred source with unchanged behavior.

```json
{"type": "is_footer", "text": "Report title", "section_matching": "nested-v1"}
```

The local ParseBench-2 v2.02 dataset opts in on its header/footer rules. Earlier dataset snapshots remain unchanged. New-mode rule results include `details.section_matching` for auditability. Pin both the dataset revision and scorer revision for comparisons; a pre-fix scorer must not be used to evaluate opted-in rules. For GT containing `<sub_page>`, dataset metadata must also set `sub_page_handling: "transparent-v1"` (CLI 2.0.3+). This strips only the container tags before text and formatting scoring, preserves all content inside, and does not require providers to emit the wrappers. Without that metadata, text scoring is unchanged.

Generation: `markdownToCombinedRules(md, {sectionMatching: "nested-v1"})` in the annotation tools forwards the contract to the generated header/footer rules. The default generator output omits the flag.
