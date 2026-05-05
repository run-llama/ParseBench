# Visual Grounding Viewer

Web app for browsing ParseBench result folders and inspecting visual grounding overlays on PDFs and images.

## Security Model

This is a local, unauthenticated file browser and result viewer. Run it on trusted machines and keep the default localhost binding unless you add your own authentication, authorization, and network hardening.

The app is intentionally self-contained under `apps/visual_grounding_viewer`:

- FastAPI backend in `backend/`
- React/Vite frontend in `frontend/`
- app-local Python dependencies in `pyproject.toml`
- app-local frontend dependencies in `frontend/package.json`

## Run In Development

```bash
cd apps/visual_grounding_viewer
./start.sh --dev
```

Dev mode starts the backend and Vite frontend separately. By default, the backend binds to `127.0.0.1:8011` and the frontend binds to `127.0.0.1:5173`.

## Run Single-Service Mode

```bash
cd apps/visual_grounding_viewer
./start.sh
```

Single-service mode installs frontend dependencies, builds `frontend/dist`, syncs Python dependencies, and serves the built frontend through Uvicorn.

## Useful Configuration

- `VISUAL_GROUNDING_VIEWER_HOST`: bind host for single-service mode.
- `VISUAL_GROUNDING_VIEWER_PORT`: bind port for single-service mode.
- `VISUAL_GROUNDING_VIEWER_DEV_BACKEND_HOST`: backend host in dev mode.
- `VISUAL_GROUNDING_VIEWER_DEV_BACKEND_PORT`: backend port in dev mode.
- `VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_HOST`: frontend host in dev mode.
- `VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_PORT`: frontend port in dev mode.
- `VITE_API_BASE_URL`: frontend API base URL in dev mode. The dev frontend falls back to `http://127.0.0.1:8011` when this is unset, so set it when using a non-default dev backend host or port.
- `VISUAL_GROUNDING_VIEWER_EXTRA_CORS_ORIGINS`: comma-separated extra CORS origins.
- `VISUAL_GROUNDING_VIEWER_BROWSE_ROOTS`: comma-separated filesystem roots exposed by the folder browser. When unset, the app uses broad local defaults such as the current home directory, `/home`, `/Users`, `/mnt`, and `/tmp` when those paths exist.
- `VISUAL_GROUNDING_VIEWER_TEST_CASE_BASE_HINTS`: comma-separated roots used to resolve test-case files when metadata contains paths from another machine.
- `VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT`: local filesystem root used to map `/files/...` URLs back to host paths.
- `VISUAL_GROUNDING_VIEWER_FILES_URL_HOSTS`: comma-separated allowed hosts for `/files/...` URL mapping. Use `*` only for trusted local workflows.
- `VISUAL_GROUNDING_VIEWER_FILES_URL_BASE_URL`: base URL used when converting host paths back to `/files/...` URLs.

## Verification

```bash
cd apps/visual_grounding_viewer
uv run pytest

cd frontend
npm ci
npm test
npm run build
```
