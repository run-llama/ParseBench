from __future__ import annotations

SOURCE_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".gif": "image",
}

ARTIFACT_SUFFIXES: dict[str, str] = {
    "v2_items": ".v2.items.json",
    "raw": ".raw.json",
    "result": ".result.json",
}

DEFAULT_PAGE_SIZE = 5000
MAX_PAGE_SIZE = 10000
