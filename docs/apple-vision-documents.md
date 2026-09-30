# Apple Vision Documents

Local macOS pipeline `apple_vision_documents`, based on
[`RecognizeDocumentsRequest`](https://developer.apple.com/documentation/vision/recognizedocumentsrequest).
Requires macOS 26+, a matching Swift SDK, and the existing `local` or `runners`
Python extra. No API key or cloud service is used.

## Build and run the test dataset

From the repository root:

```sh
mkdir -p .build
xcrun swiftc -O src/parse_bench/apple_vision_documents.swift -o .build/apple-vision-documents
uv sync --extra runners --extra dev
uv run parse-bench inference run apple_vision_documents --input_dir data/test --output_dir output/apple-vision-test --max_concurrent 1 --no_rich --timeout_retries 0
uv run parse-bench evaluation run output/apple-vision-test --test_cases_dir data/test --max_workers 1
uv run parse-bench evaluation run output/apple-vision-test --test_cases_dir data/test --group text_content --report_dir output/apple-vision-text-content --max_workers 1
```

These commands use only the already downloaded test dataset. Inference and
evaluation are separate so results can be inspected without repeating OCR.
The extra text-content evaluation works around
[ParseBench issue #173](https://github.com/run-llama/ParseBench/issues/173):
unfiltered evaluation overwrites text-content cases with text-formatting cases
that share an inference ID. Keep both reports; do not treat the unfiltered
report alone as complete coverage of all five categories.

The shorter end-to-end command (still requiring the separate content evaluation) is:

```sh
uv run parse-bench run apple_vision_documents --input_dir data/test --output_dir output/apple-vision-test --max_concurrent 1 --open_report=False
```

Test the bridge directly with any page image:

```sh
.build/apple-vision-documents /absolute/path/to/page.png > page.json
```

The provider checks `APPLE_VISION_DOCUMENTS_BIN`, then the executable on `PATH`,
then `.build/apple-vision-documents` in the current directory. A pipeline config
`binary` overrides those choices. Build explicitly before benchmarking; build
time is excluded. The Python wheel includes the Swift source. To locate it from
an installed package, run:

```sh
uv run python -c 'from importlib.resources import files; print(files("parse_bench") / "apple_vision_documents.swift")'
```

## Output and timing

- Render each PDF page at 200 DPI in RGB with PyMuPDF. Invoke one Swift process
  per PDF and run its pages sequentially with a fresh Vision request per page.
  Use `--max_concurrent 1` for comparable local timings.
- Pin Vision request revision 1 and retain its OS version in every raw page
  result. OS/model changes can affect scores.
- Emit JSON to stdout and errors to stderr. Missing input, recognition errors,
  malformed JSON, and an aggregate timeout of 120 seconds per page fail the
  document. A successful blank page remains empty. `dpi` and `page_timeout` are
  pipeline settings.
- Preserve raw paragraphs, tables, cell text/ranges/spans, lists, and detected
  titles. Boxes are normalized to `[0, 1]` with a top-left origin. Vision's
  bounding regions become axis-aligned enclosing rectangles.
- Normalize detected titles to level-one Markdown headings, paragraphs to
  Markdown, and tables to HTML, including `rowspan` and `colspan`. Table metrics
  require HTML. Line, word, and cell boxes populate ParseBench's granular
  grounding layers; page/block boxes feed the layout adapter.
- Keep Vision's paragraph order. Replace paragraphs whose centers fall inside
  a table or list with that region once. Regions not matched to a paragraph are
  inserted by vertical position. This fallback can misorder columns; Apple also
  makes reading-order errors. List markers are retained. Detected document titles
  become level-one headings; deeper heading levels and table headers are not guessed.
- Raw `pages[].recognition_latency_ms` measures the Vision request only.
  `render_latency_ms` measures rasterization and PNG writing.
  Per-page `latency_in_ms` is render plus recognition time. The result's top-level
  latency also includes process launch, JSON decoding, PDF opening, and cleanup,
  but excludes normalization/evaluation. There is no warmup, and local compute
  cost is not assigned a dollar price.

This first integration does not infer chart data, formulas, styling, or semantic
labels beyond title, text, table, and list regions. Test-subset scores are smoke-test
results, not comparable to the full ParseBench leaderboard.

## Checks

```sh
uv run --extra runners --extra dev pytest -q
uv run --extra runners --extra dev ruff check src/parse_bench/inference/providers/parse/apple_vision_documents.py tests/parse_bench/inference/providers/parse/test_apple_vision_documents.py
```

The provider tests cover table placement, spans, HTML escaping, blank pages,
layout-coordinate scaling, latency retention, malformed JSON, CLI errors, and
timeouts. The real test-dataset run checks the Swift/API boundary.
