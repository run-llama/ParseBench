Integrate a new document parsing pipeline into ParseBench: $ARGUMENTS

You are integrating a new pipeline into the ParseBench benchmark. The user will provide a pipeline name and any relevant context (API docs, SDK links, product website, etc.). Your job is to create all the files needed so that `uv run parse-bench run <pipeline_name>` works end-to-end.

---

## Step 1: Understand the provider

Before writing any code, research the provider:

1. If the user gave a URL, fetch and read it to understand the API/SDK.
2. Determine:
   - **Product type**: Is this a `PARSE` provider (PDF -> markdown) or `LAYOUT_DETECTION` provider (PDF -> bounding boxes)?
   - **Integration style**: Cloud API (needs API key), self-hosted model (needs endpoint URL), or local library (no external deps)?
   - **SDK/API pattern**: Does it have a Python SDK? REST API? What's the auth method?
   - **Input format**: Does it accept PDF files directly, or does it need images (page screenshots)?
   - **Output format**: What does the raw response look like? Markdown? HTML? JSON with pages?

---

## Step 2: Find the closest existing provider to use as a template

Look at the existing providers and pick the best template:

- **Cloud API with Python SDK** (e.g., OpenAI, Anthropic, Google): Copy from `src/parse_bench/inference/providers/parse/openai.py` or `anthropic_haiku.py`
- **Cloud API with REST calls**: Copy from `src/parse_bench/inference/providers/parse/reducto.py` or `chunkr.py`
- **Self-hosted vLLM endpoint**: Copy from `src/parse_bench/inference/providers/parse/gemma4.py` or `qwen3_5.py`
- **Local library (no API)**: Copy from `src/parse_bench/inference/providers/parse/pymupdf.py` or `tesseract.py`
- **Layout detection**: Copy from `src/parse_bench/inference/providers/layoutdet/docling.py`

Read the template file to understand the exact pattern.

---

## Step 3: Create the provider file

Create `src/parse_bench/inference/providers/parse/<provider_name>.py` (or `layoutdet/` for layout detection).

The provider must:

1. Import and use the `@register_provider("<provider_name>")` decorator from `parse_bench.inference.providers.registry`
2. Subclass `Provider` from `parse_bench.inference.providers.base`
3. Implement `__init__`, `run_inference`, and `normalize`:

```python
from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType


@register_provider("<provider_name>")
class MyProvider(Provider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        # Read config from self.base_config
        # Read API keys from os.environ
        # Import SDK lazily (inside __init__, not at module level)
        # Raise ProviderConfigError for missing keys/deps

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        # Validate request.product_type
        # Call the external API/SDK
        # Return RawInferenceResult with raw_output as a dict
        # Use ProviderTransientError for retryable errors (network, rate limits)
        # Use ProviderPermanentError for non-retryable errors (bad file, 4xx)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        # Convert raw_output dict into ParseOutput (or LayoutOutput)
        # ParseOutput needs: task_type="parse", example_id, pipeline_name, pages=[PageIR(...)], markdown=full_text
        # Return InferenceResult wrapping both raw and normalized output
```

Key conventions:
- Import SDKs lazily (inside `__init__` or methods, not at module top level) to avoid dependency issues
- API keys come from `os.environ`, not from config dict
- Config options (model, timeout, dpi, mode, etc.) come from `self.base_config`
- Error classification: network/timeout/5xx/rate-limit -> `ProviderTransientError`; bad input/4xx -> `ProviderPermanentError`; missing config -> `ProviderConfigError`
- For vision-based providers that need page images: use `pdf2image.convert_from_path()` with configurable DPI
- `raw_output` should preserve the full API response for debugging
- `normalize()` must produce `ParseOutput` with `pages: list[PageIR]` and `markdown: str` (concatenated page markdowns)

---

## Step 4: Register the provider module

Add the module name to the `_PROVIDER_MODULES` list in:
- `src/parse_bench/inference/providers/parse/__init__.py` (for parse providers)
- Or add an import in `src/parse_bench/inference/providers/layoutdet/__init__.py` (for layout providers)

The list is alphabetically sorted. The module name is just the filename without `.py`.

---

## Step 5: Register pipeline configurations

Add pipeline definitions to:
- `src/parse_bench/inference/pipelines/parse.py` → inside `register_parse_pipelines()`
- Or `src/parse_bench/inference/pipelines/layout.py` → inside `register_layout_pipelines()`

Each pipeline is a `PipelineSpec`:

```python
register_fn(
    PipelineSpec(
        pipeline_name="<provider>_<variant>",    # e.g., "acme_fast", "acme_accurate"
        provider_name="<provider_name>",          # Must match @register_provider name
        product_type=ProductType.PARSE,            # or LAYOUT_DETECTION
        config={                                   # Passed to Provider.__init__ as base_config
            "model": "acme-v2",
            "timeout": 120,
        },
    )
)
```

Naming conventions:
- Pipeline names: `{provider}_{variant}` (e.g., `openai_gpt5_mini_parse`, `reducto_agentic_chart`)
- Add a comment section header for the new provider (see existing examples in the file)
- Register multiple variants if the provider has different modes/tiers

---

## Step 6: Update documentation

Add the new pipeline(s) to `docs/pipelines.md` under the appropriate section (Cloud API / Self-hosted / Local).

Use the existing table format:

```markdown
### Provider Name

| Pipeline | Description | Env Var |
|---|---|---|
| `pipeline_name` | Short description | `ENV_VAR_NAME` |
```

---

## Step 7: Verify

Run these commands to verify the integration:

```bash
# Check the pipeline appears in the list
uv run parse-bench pipelines

# Dry-run test on a single file (if the user has API access)
uv run parse-bench run <pipeline_name> --test
```

If there are import errors or missing dependencies, fix them. The lazy import pattern in `parse/__init__.py` means missing optional deps won't crash the whole system — they'll just skip that provider.

---

## Summary checklist

- [ ] Provider file created in `providers/parse/` or `providers/layoutdet/`
- [ ] Provider registered with `@register_provider()` decorator
- [ ] Module added to `_PROVIDER_MODULES` list in `__init__.py`
- [ ] Pipeline(s) registered in `pipelines/parse.py` or `pipelines/layout.py`
- [ ] `docs/pipelines.md` updated with new pipeline entries
- [ ] `uv run parse-bench pipelines` shows the new pipeline(s)
