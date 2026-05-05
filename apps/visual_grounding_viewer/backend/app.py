from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import HealthResponse
from .routes.browse import router as browse_router
from .routes.document import router as document_router
from .routes.index import router as index_router

app = FastAPI(title="Visual Grounding Viewer", version="0.1.0")


def _allowed_origins() -> list[str]:
    origins: list[str] = []
    for port in range(5173, 5181):
        origins.append(f"http://localhost:{port}")
        origins.append(f"http://127.0.0.1:{port}")
    extra_origins = os.getenv("VISUAL_GROUNDING_VIEWER_EXTRA_CORS_ORIGINS", "")

    for raw_origin in extra_origins.split(","):
        origin = raw_origin.strip()
        if origin and origin not in origins:
            origins.append(origin)

    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


app.include_router(index_router)
app.include_router(document_router)
app.include_router(browse_router)
