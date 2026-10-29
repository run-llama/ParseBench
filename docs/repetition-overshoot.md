# Repetition overshoot

Some pages make a vision model lose its place: a dot leader that never ends, a grid of identical cells, a code
written in O and 0, a script it has barely seen. The model keeps going, one more dot, one more row, and writes more
than the page holds. `repetition_overshoot` is a built-in rule that catches that, and only that.

## Running it on your own documents

```bash
uv run parse-bench repetition make ./my_docs     # writes <name>.test.json next to each PDF
uv run parse-bench run <pipeline_name> --input_dir ./my_docs
```

`repetition make` reads each PDF's text layer (with PyMuPDF: `pip install 'parse-bench[local]'`) and writes one
`repetition_overshoot` rule budgeted from it. A PDF without a text layer is skipped, since there is nothing to budget
from; `--scans` tests it instead as a page that repeats nothing (any unit 20 copies, fill 40 characters, 5000 letters
and digits). Existing tests are kept unless you pass `--overwrite`.

## Running a dataset from HuggingFace

`--dataset` on `run`, `download` and `status` takes `parsebench` (the default) or the id of any HuggingFace dataset
repo in the ParseBench layout (one `<category>.jsonl` of rows per category and the documents under `docs/`). It
downloads to `./data/<name, lowercased>`, and `--group` picks one category:

```bash
uv run parse-bench run <pipeline_name> --dataset org/name
uv run parse-bench run <pipeline_name> --dataset org/name --group <category>
uv run parse-bench download --dataset org/name
uv run parse-bench status --dataset org/name
```

A private repo needs `hf auth login` or `HF_TOKEN`. Such a dataset has no `--test` subset.

## The rule

The headline number is the rule pass rate: the share of pages with no overshoot. A page the parser returns empty
fails, so a parser cannot pass by saying nothing.

`repetition_overshoot` cleans the output of markdown and HTML syntax (table separators, grids, rules, bold markers,
tags), folds look-alike characters (`O`/`0`, `I`/`l`/`ǀ`/`1`, `$`/`5`) so a misread is not a loop, then fails on the
worst overshoot of:

| Budget | Limit |
|---|---|
| A fill run (dots, dashes, underscores, `=`, `·`, `…`, `▪`) | 2 × the page's longest + 40, one line of the page (250), or 2 × the page's block of consecutive leader lines + 40 |
| All fill together | 2 × the page's total + 40, or 250 per line of the page |
| A run of one character, a 2-8 character unit or a token | 2 × its count on the page + 20 |
| A repeated line | 2 × the count of its rarest word on the page + 40 (the line's own count on an exact page) |
| A whole table written out again | 2 × its count on the page (at least once) |
| An HTML table row repeated | as a repeated line |

On a text-layer page, a unit the text layer lacks (a value drawn in a chart) is not budgeted, and a value repeated
across a table's complete, distinct rows (a chart transcribed point by point) is data, not a repeated line. On an
exact page, every copy of a unit counts, not only its longest run.

Restyling, splitting, merging or dropping a repetition never fails. The budgets are loose on purpose: a text layer can
undercount a leader drawn with a symbol font, so one line of the page is always allowed.

## Known limitations

- A text layer is not a human transcription. A garbled text layer can undercount a value the page shows many times.
- With the default budgets, a loop shorter than the page's own content of that unit is allowed (the factor is 2).
