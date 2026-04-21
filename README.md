# ParseBench

[![Website](https://img.shields.io/badge/Website-parsebench.ai-blue)](https://parsebench.ai)
[![arXiv](https://img.shields.io/badge/arXiv-2604.08538-b31b1b.svg)](https://arxiv.org/abs/2604.08538)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/llamaindex/ParseBench)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**ParseBench** is a benchmark for evaluating how well document parsing tools convert PDFs into structured output that AI agents can reliably act on. It tests whether parsed output preserves the structure and meaning needed for autonomous decisions — not just whether it looks similar to a reference text.

The benchmark covers ~2,000 human-verified pages from real enterprise documents (insurance, finance, government), organized around five capability dimensions, each targeting a failure mode that breaks production agent workflows.

<p align="center">
  <img src="docs/parsebench_teaser.png" alt="ParseBench overview: five capability dimensions" width="100%">
</p>

## Leaderboard

<!-- LEADERBOARD:START -->
_Top 10 by Overall score. For the full sortable, filterable leaderboard, see [parsebench.ai](https://parsebench.ai/); for raw data, see [leaderboard.csv](leaderboard.csv)._

| Rank | Provider | Category | Overall | Tables | Charts | Content Faith. | Sem. Format. | Visual Ground. | ¢ / Page |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | LlamaParse Agentic | LlamaParse | 84.88 | 90.74 | 78.11 | 89.68 | 85.24 | 80.62 | 1.25¢ |
| 2 | Google Gemini 3 Flash (Thinking High) | VLM - Proprietary | 75.05 | 91.50 | 64.79 | 90.87 | 68.31 | 59.77 | 2.41¢ |
| 3 | Reducto (Agentic) | Commercial - Startup APIs | 72.97 | 80.42 | 73.4 | 86.37 | 57.6 | 67.07 | 4.76¢ |
| 4 | LlamaParse Cost Effective | LlamaParse | 71.89 | 73.16 | 66.66 | 88.02 | 73.04 | 58.56 | 0.38¢ |
| 5 | Google Gemini 3 Flash (Thinking Minimal) | VLM - Proprietary | 71.04 | 89.85 | 64.83 | 86.19 | 58.35 | 55.97 | 0.65¢ |
| 6 | Chandra-ocr-2 | VLM - Open Weight | 70.1 | 89.2 | 65.1 | 83.7 | 61.4 | 51.2 | — |
| 7 | Google Gemini 3.1 Pro | VLM - Proprietary | 69.14 | 91.00 | 41.13 | 90.16 | 52.43 | 70.99 | 8.49¢ |
| 8 | Reducto | Commercial - Startup APIs | 67.83 | 70.33 | 56.99 | 86.37 | 56.75 | 68.71 | 2.38¢ |
| 9 | Extend (Beta) | Commercial - Startup APIs | 67.83 | 85.93 | 40.42 | 85.03 | 59.49 | 68.28 | 2.50¢ |
| 10 | Anthropic Opus 4.7 | VLM - Proprietary | 63.34 | 87.17 | 55.84 | 90.26 | 69.42 | 13.99 | 7.14¢ |
<!-- LEADERBOARD:END -->

## Quick Start

**Prerequisites:** Create a `.env` file with the API key for the parsing tool you want to evaluate (see [Configuration](#configuration) for details).

```bash
# Install
uv sync --extra runners

# Quick test run (small dataset, 3 files per category — good for trying things out)
uv run parse-bench run llamaparse_agentic --test

# Full benchmark run (replace llamaparse_agentic with any pipeline name, see "Available Pipelines" below)
uv run parse-bench run llamaparse_agentic

# View interactive reports in your browser
uv run parse-bench serve llamaparse_agentic
```

## Available Pipelines

A **pipeline** is a document parsing tool or configuration that you want to evaluate. There are 90+ pipelines available -- see [docs/pipelines.md](docs/pipelines.md) for the full list, or run `uv run parse-bench pipelines`.

<details>
<summary><strong>Paper baselines (21 pipelines)</strong></summary>

| Pipeline name | Name in paper |
|---------------|---------------|
| `llamaparse_agentic` | LlamaParse Agentic |
| `llamaparse_cost_effective` | LlamaParse Cost Effective |
| `openai_gpt5_mini_reasoning_medium_parse_with_layout_file` | OpenAI GPT-5 Mini (Reasoning Medium) |
| `openai_gpt5_mini_reasoning_minimal_parse_with_layout_file` | OpenAI GPT-5 Mini (Reasoning Minimal) |
| `openai_gpt_5_4_parse_with_layout_file` | OpenAI GPT-5.4 |
| `anthropic_haiku_parse_with_layout_file` | Anthropic Haiku 4.5 (Disable Thinking) |
| `anthropic_haiku_thinking_parse_with_layout_file` | Anthropic Haiku 4.5 (Thinking) |
| `anthropic_opus_4_6_parse_with_layout_file` | Anthropic Opus 4.6 |
| `google_gemini_3_flash_thinking_minimal_parse_with_layout_file` | Google Gemini 3 Flash (Thinking Minimal) |
| `google_gemini_3_flash_thinking_high_parse_with_layout_file` | Google Gemini 3 Flash (Thinking High) |
| `google_gemini_3_1_pro_parse_with_layout_file` | Google Gemini 3.1 Pro |
| `azure_di_layout` | Azure Document Intelligence |
| `aws_textract` | AWS Textract |
| `google_docai_layout` | Google Cloud Document AI |
| `reducto` | Reducto |
| `reducto_agentic` | Reducto (Agentic) |
| `extend_parse` | Extend |
| `landingai_parse` | LandingAI |
| `qwen3_5_4b_vllm_parse` | Qwen 3 VL |
| `dots_ocr_1_5_parse` | Dots OCR 1.5 |
| `docling_parse` | Docling |

</details>

## Dataset

Hosted on HuggingFace: [`llamaindex/ParseBench`](https://huggingface.co/datasets/llamaindex/ParseBench)

The dataset is stratified into five capability dimensions, each with its own ground-truth format and evaluation metric:

| Dimension | File(s) | Metric | Pages | Docs | Rules |
|-----------|---------|--------|------:|-----:|------:|
| **Tables** | `table.jsonl` | GTRM (GriTS + TableRecordMatch) | 503 | 284 | --- |
| **Charts** | `chart.jsonl` | ChartDataPointMatch | 568 | 99 | 4,864 |
| **Content Faithfulness** | `text_content.jsonl` | Content Faithfulness Score | 506 | 506 | 141,322 |
| **Semantic Formatting** | `text_formatting.jsonl` | Semantic Formatting Score | 476 | 476 | 5,997 |
| **Visual Grounding** | `layout.jsonl` | Element Pass Rate | 500 | 321 | 16,325 |
| **Total (unique)** | | | **2,078** | **1,211** | **169,011** |

Content Faithfulness and Semantic Formatting share the same 507 underlying text documents, evaluated with different rule sets. Totals reflect unique pages and documents. Tables uses a continuous metric (no discrete rules).

**What each dimension tests and why it matters for agents:**

- **Tables** — Structural fidelity of merged cells and hierarchical headers. A misaligned header means the agent reads the wrong column when looking up a value.
- **Charts** — Exact data point extraction with correct series and axis labels from bar, line, pie, and compound charts. Most parsers return raw text instead of structured data, leaving agents unable to extract precise values.
- **Content Faithfulness** — Omissions, hallucinations, and reading-order violations. If the agent's context is incomplete or contains fabricated content, every downstream decision is compromised.
- **Semantic Formatting** — Preservation of formatting that carries meaning: strikethrough (marks superseded content), superscript/subscript (footnotes, formulas), bold (defined terms, key values), and title hierarchy. A strikethrough price is not the current price.
- **Visual Grounding** — Tracing every extracted element back to its source location on the page. Required for auditability in regulated workflows where every value must be traceable.

The dataset is automatically downloaded when you run a pipeline. To manage it manually:

```bash
# Download the full dataset
uv run parse-bench download

# Download a small test dataset (3 files per category, good for trying things out)
uv run parse-bench download --test

# Check whether the dataset has been downloaded and show summary statistics
uv run parse-bench status
```

## Usage

### Running the Benchmark

The `run` command runs inference (calls the parsing tool), evaluates the results against ground truth, and generates reports:

```bash
# Evaluate a parsing tool on all five dimensions
uv run parse-bench run <pipeline_name>

# Evaluate on a single dimension only (e.g., chart, table, layout, text_content, text_formatting)
uv run parse-bench run <pipeline_name> --group chart

# Skip calling the parsing tool — just re-evaluate existing results
uv run parse-bench run <pipeline_name> --skip_inference

# Control how many pages are processed in parallel
uv run parse-bench run <pipeline_name> --max_concurrent 10

# Run on the small test dataset only (3 files per category, good for trying things out)
uv run parse-bench run <pipeline_name> --test
```

When running all dimensions, the benchmark produces:
- Per-dimension detailed HTML reports with drill-down per test case
- An aggregation dashboard showing all dimensions side-by-side
- A leaderboard comparing all evaluated tools in the output directory
- CSV, Markdown, and JSON exports per dimension

### Viewing & Comparing Results

```bash
# View reports in your browser (needed because browsers block PDF rendering from file:// URLs)
uv run parse-bench serve <pipeline_name>

# Compare two parsing tools side-by-side
uv run parse-bench compare <pipeline_a> <pipeline_b>

# Generate a leaderboard across all evaluated tools
uv run parse-bench leaderboard

# Leaderboard for specific tools only
uv run parse-bench leaderboard llamaparse_agentic llamaparse_cost_effective
```

<details>
<summary><strong>Advanced Subcommands</strong></summary>

For fine-grained control over individual steps:

```bash
# Run inference only (call the parsing tool, don't evaluate)
uv run parse-bench inference run <pipeline_name>

# Run evaluation only (on existing inference results)
uv run parse-bench evaluation run --output_dir ./output/<pipeline_name>

# Generate detailed HTML report from evaluation results
uv run parse-bench analysis generate_report --evaluation_dir ./output/<pipeline_name>

# Regenerate the aggregation dashboard
uv run parse-bench analysis generate_dashboard --evaluation_dir ./output/<pipeline_name>
```

</details>

<details>
<summary><strong>Evaluating Your Own Tool</strong></summary>

To add a new parsing tool to ParseBench, use [Claude Code](https://claude.ai/code):

```bash
/integrate-pipeline <name> <API docs or SDK link>
```

This creates the provider, registers the pipeline, and updates docs. The skill definition lives in [`.claude/commands/integrate-pipeline.md`](.claude/commands/integrate-pipeline.md) and can be adapted for other AI coding agents.

</details>

## Configuration

### API Keys

Each pipeline calls a specific parsing tool's API. You only need the API key for the tool you want to evaluate — add it to a `.env` file at the project root:

```bash
# Only add the keys you need. For example, to evaluate LlamaParse:
LLAMA_CLOUD_API_KEY=...

# To evaluate OpenAI-based pipelines:
OPENAI_API_KEY=...

# To evaluate Anthropic-based pipelines:
ANTHROPIC_API_KEY=...

# To evaluate Google-based pipelines:
GOOGLE_API_KEY=...
```

ParseBench does **not** use LLM-as-a-judge — all evaluation is deterministic and rule-based. API keys are only used to call the parsing tool being evaluated.

### CLI Reference

| Command | Description |
|---------|-------------|
| `parse-bench run` | Evaluate a parsing tool end-to-end (inference + evaluation + reports) |
| `parse-bench download` | Download the benchmark dataset from HuggingFace |
| `parse-bench status` | Check whether the dataset has been downloaded |
| `parse-bench pipelines` | List all available parsing tools / pipeline configurations |
| `parse-bench compare` | Compare results from two parsing tools side-by-side |
| `parse-bench leaderboard` | Generate a leaderboard across all evaluated tools |
| `parse-bench serve` | View HTML reports in your browser (with PDF rendering support) |

Advanced subcommands: `inference`, `evaluation`, `analysis`, `pipeline`, `data`

<details>
<summary><strong>Output Structure</strong></summary>

```
output/
├── _leaderboard.html                       # Cross-pipeline leaderboard
└── <pipeline_name>/
    ├── chart/
    │   ├── *.result.json                    # Inference results
    │   ├── _evaluation_report.json          # Evaluation summary
    │   ├── _evaluation_report_detailed.html # Interactive detailed report
    │   ├── _evaluation_results.csv          # Per-example CSV
    │   └── _evaluation_report.md            # Markdown summary
    ├── layout/   (same structure)
    ├── table/    (same structure)
    ├── text_content/   (same structure)
    ├── text_formatting/ (same structure)
    ├── _evaluation_report_dashboard.html    # Aggregation dashboard
    └── _metadata.json                       # Run metadata
```

</details>

<details>
<summary><strong>Project Structure</strong></summary>

```
src/parse_bench/
├── cli.py                           # Fire CLI entry point
├── pipeline/cli.py                  # End-to-end pipeline orchestration
├── data/
│   ├── download.py                  # HuggingFace dataset download
│   └── cli.py                       # Data management CLI
├── inference/
│   ├── runner.py                    # Batch inference with concurrency
│   ├── pipelines/                   # Pipeline registry (parse, extract, layout)
│   └── providers/                   # Provider implementations per product type
├── evaluation/
│   ├── runner.py                    # Parallel evaluation
│   ├── evaluators/                  # Product-specific evaluators (parse, extract, layout)
│   ├── metrics/                     # Metric implementations (TEDS, GriTS, rules, IoU)
│   └── reports/                     # CSV, HTML, markdown export
├── analysis/
│   ├── aggregation_report.py        # Multi-category dashboard
│   ├── detailed_report.py           # Interactive per-category HTML report
│   ├── comparison.py                # Pipeline comparison
│   └── comparison_report.py         # Comparison HTML report
├── test_cases/
│   ├── loader.py                    # Load test cases (JSONL or sidecar .test.json)
│   └── schema.py                    # TestCase types (Parse, Extract, LayoutDetection)
└── schemas/
    ├── pipeline_io.py               # InferenceRequest, InferenceResult
    ├── evaluation.py                # EvaluationResult, EvaluationSummary
    └── product.py                   # ProductType enum (PARSE, EXTRACT, LAYOUT_DETECTION)
```

</details>

## Citation

```bibtex
@misc{zhang2026parsebench,
  title={ParseBench: A Document Parsing Benchmark for AI Agents},
  author={Boyang Zhang and Sebastián G. Acosta and Preston Carlson and Sacha Bron and Pierre-Loïc Doulcet and Daniel B. Ospina and Simon Suo},
  year={2026},
  eprint={2604.08538},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.08538},
}
```

## Links

- **Paper**: [arXiv:2604.08538](https://arxiv.org/abs/2604.08538)
- **HuggingFace Dataset**: [llamaindex/ParseBench](https://huggingface.co/datasets/llamaindex/ParseBench)
- **Code**: [run-llama/ParseBench](https://github.com/run-llama/ParseBench)
