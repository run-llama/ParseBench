from __future__ import annotations

from io import BytesIO
from functools import lru_cache
import fitz
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

from ..loader import load_document
from ..models import DocumentResponse
from ..state import STATE

router = APIRouter(prefix="/api", tags=["document"])


def _resolve_doc(session_id: str, doc_id: str):
    session = STATE.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

    doc = session.docs_by_id.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")

    return doc


@lru_cache(maxsize=512)
def _render_pdf_page(path_str: str, page_index: int, mtime_ns: int) -> bytes:
    del mtime_ns
    with fitz.open(path_str) as doc:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(f"Page out of range: {page_index}")
        page = doc.load_page(page_index)
        pix = page.get_pixmap(alpha=False, dpi=144)
        return pix.tobytes("png")


@lru_cache(maxsize=512)
def _render_image_source(path_str: str, mtime_ns: int) -> bytes:
    del mtime_ns
    with Image.open(path_str) as image:
        rendered = image.convert("RGB")
        buffer = BytesIO()
        rendered.save(buffer, format="PNG")
        return buffer.getvalue()


@router.get("/document", response_model=DocumentResponse)
def get_document(
    session_id: str = Query(...),
    doc_id: str = Query(...),
) -> DocumentResponse:
    doc = _resolve_doc(session_id, doc_id)
    return load_document(doc)


@router.get("/source_asset")
def get_source_asset(
    session_id: str = Query(...),
    doc_id: str = Query(...),
):
    doc = _resolve_doc(session_id, doc_id)
    return FileResponse(path=doc.source_path)


@router.get("/page_asset")
def get_page_asset(
    session_id: str = Query(...),
    doc_id: str = Query(...),
    page: int = Query(default=1, ge=1),
):
    doc = _resolve_doc(session_id, doc_id)
    source_path = doc.source_path

    if doc.source_kind == "image":
        if page != 1:
            raise HTTPException(status_code=400, detail="Image sources only have page=1")
        page_bytes = _render_image_source(
            str(source_path),
            source_path.stat().st_mtime_ns,
        )
        return Response(content=page_bytes, media_type="image/png")

    page_index = page - 1
    try:
        page_bytes = _render_pdf_page(
            str(source_path),
            page_index,
            source_path.stat().st_mtime_ns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=page_bytes, media_type="image/png")
