from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..indexer import build_index
from ..models import IndexRequest, IndexResponse
from ..state import STATE

router = APIRouter(prefix="/api", tags=["index"])


@router.post("/index", response_model=IndexResponse)
def post_index(request: IndexRequest) -> IndexResponse:
    try:
        result = build_index(
            root_path=request.root_path,
            test_cases_path=request.test_cases_path,
            page=request.page,
            page_size=request.page_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = STATE.create_session(
        root_path=Path(result.response.resolved_root_path),
        docs_by_id=result.docs_by_id,
    )

    return result.response.model_copy(update={"session_id": session_id})
