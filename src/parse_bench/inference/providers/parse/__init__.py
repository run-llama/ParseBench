"""Parse providers — imported lazily to avoid requiring all SDKs."""

import importlib
import logging

logger = logging.getLogger(__name__)

_PROVIDER_MODULES = [
    "anthropic",
    "azure_document_intelligence",
    "chandra2",
    "chunkr",
    "databricks_ai_parse",
    "datalab",
    "deepseekocr2",
    "docling",
    "docling_serve",
    "dots_ocr",
    "extend_parse",
    "falconocr",
    "gemma4",
    "google",
    "google_docai",
    "granite_vision",
    "infinity_parser2",
    "kdl_frontier_nano",
    "landingai",
    "liteparse",
    "markitdown",
    "opendataloader",
    "pdf_inspector",
    "pymupdf4llm",
    "llamaparse",
    "llamaparse_v2_normalization",
    "mineru25",
    "mineru2605pro",
    "mistral_ocr",
    "openai",
    "paddleocr",
    "pulse",
    "pymupdf",
    "pypdf",
    "qwen3_5",
    "reducto",
    "surya2",
    "tesseract",
    "textract",
    "unlimitedocr",
    "unstructured",
]

for _mod in _PROVIDER_MODULES:
    try:
        importlib.import_module(f"parse_bench.inference.providers.parse.{_mod}")
    except ImportError:
        logger.debug("Skipping parse provider %s (missing dependency)", _mod)
