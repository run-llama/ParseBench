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
    "dots_ocr",
    "extend_parse",
    "gemma4",
    "google",
    "google_docai",
    "granite_vision",
    "landingai",
    "llamaparse",
    "llamaparse_v2_normalization",
    "mineru25",
    "openai",
    "paddleocr",
    "pymupdf",
    "pypdf",
    "qwen3_5",
    "reducto",
    "tesseract",
    "textract",
    "unstructured",
]

for _mod in _PROVIDER_MODULES:
    try:
        importlib.import_module(f"parse_bench.inference.providers.parse.{_mod}")
    except ImportError:
        logger.debug("Skipping parse provider %s (missing dependency)", _mod)
