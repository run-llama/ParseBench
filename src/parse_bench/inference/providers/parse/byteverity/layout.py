"""Layout evidence from the open-source docling-layout-heron detector (Apache-2.0), CPU.

Runs in-process when torch+transformers import; otherwise in a worker under BYTEVERITY_VISION_PY.
Returns [{"label","score","bbox"(0-1000)}] per page. Evidence only: oracle O14 decides whether it is used.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import threading
from typing import Any

REPO = "docling-project/docling-layout-heron"
_LOCK = threading.Lock()
_MODEL: Any = None


def _inprocess():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            proc = AutoImageProcessor.from_pretrained(REPO)
            model = AutoModelForObjectDetection.from_pretrained(REPO).eval()
            _MODEL = (proc, model)
    return _MODEL


def detect_page(page: Any, dpi: int = 100, threshold: float = 0.3) -> list[dict[str, Any]]:
    try:
        import torch  # noqa: F401
        from PIL import Image

        proc, model = _inprocess()
    except Exception:
        return _detect_subprocess(page, dpi, threshold)
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    import torch

    with _LOCK, torch.inference_mode():
        out = model(**proc(images=img, return_tensors="pt"))
        r = proc.post_process_object_detection(out, target_sizes=[img.size[::-1]], threshold=threshold)[0]
    labels = model.config.id2label
    return [
        {
            "label": labels[int(li)],
            "score": round(float(s), 3),
            "bbox": [round(float(v) / d * 1000, 1) for v, d in zip(b, [img.width, img.height] * 2, strict=False)],
        }
        for s, li, b in zip(r["scores"], r["labels"], r["boxes"], strict=False)
    ]


def _detect_subprocess(page: Any, dpi: int, threshold: float) -> list[dict[str, Any]]:
    py = os.environ.get("BYTEVERITY_VISION_PY")
    if not py:
        return []
    pdf = page.parent.name
    # load this file by path: the vision interpreter needs only torch/transformers/pymupdf, not parse_bench
    code = (
        "import json,importlib.util,pymupdf;"
        f"s=importlib.util.spec_from_file_location('bv_layout',{os.path.abspath(__file__)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"p=pymupdf.open({pdf!r})[{page.number}];print(json.dumps(m.detect_page(p,{dpi},{threshold})))"
    )
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=600)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else []
    except Exception:
        return []
