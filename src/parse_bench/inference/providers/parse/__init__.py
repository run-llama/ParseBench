"""Parse providers — imported lazily to avoid requiring all SDKs."""

import importlib
import logging

logger = logging.getLogger(__name__)

_PROVIDER_MODULES = [
    "amazon_nova",
    "anthropic",
    "anyformat",
    "azure_document_intelligence",
    "chandra2",
    "chunkr",
    "databricks_ai_parse",
    "datalab",
    "deepseek",
    "deepseekocr2",
    "docling",
    "docling_serve",
    "dots_ocr",
    "extend_parse",
    "falconocr",
    "florin_parser_nano",
    "gemma4",
    "glm_zai",
    "google",
    "google_docai",
    "granite_vision",
    "infinity_parser2",
    "jinaocr",
    "kdl_frontier_nano",
    "landingai",
    "liteparse",
    "markitdown",
    "opendataloader",
    "pdf_inspector",
    "pymupdf4llm",
    "rakedoc_nano",
    "llamaparse",
    "llamaparse_v2_normalization",
    "mineru25",
    "mineru2605pro",
    "mineru_diffusion",
    "mistral_ocr",
    "nemotron_omni",
    "nutrient_dws",
    "openai",
    "paddleocr",
    "pulse",
    "pymupdf",
    "pypdf",
    "qwen",
    "reducto",
    "surya2",
    "tesseract",
    "textract",
    "unlimitedocr",
    "unstructured",
    "warp_ingest",
    "oi_parser",
]

for _mod in _PROVIDER_MODULES:
    try:
        importlib.import_module(f"parse_bench.inference.providers.parse.{_mod}")
    except ImportError:
        logger.debug("Skipping parse provider %s (missing dependency)", _mod)
