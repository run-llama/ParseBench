#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR"
FRONTEND_DIR="$APP_DIR/frontend"

VISUAL_GROUNDING_VIEWER_HOST="${VISUAL_GROUNDING_VIEWER_HOST:-127.0.0.1}"
VISUAL_GROUNDING_VIEWER_PORT="${VISUAL_GROUNDING_VIEWER_PORT:-8004}"
VISUAL_GROUNDING_VIEWER_DEV_BACKEND_HOST="${VISUAL_GROUNDING_VIEWER_DEV_BACKEND_HOST:-127.0.0.1}"
VISUAL_GROUNDING_VIEWER_DEV_BACKEND_PORT="${VISUAL_GROUNDING_VIEWER_DEV_BACKEND_PORT:-8011}"
VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_HOST="${VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_HOST:-127.0.0.1}"
VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_PORT="${VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_PORT:-5173}"
DEV_MODE=false
SKIP_BUILD=false

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Start the visual grounding viewer.

Options:
    --dev           Run split dev mode: backend on $VISUAL_GROUNDING_VIEWER_DEV_BACKEND_HOST:$VISUAL_GROUNDING_VIEWER_DEV_BACKEND_PORT and frontend on $VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_HOST:$VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_PORT
    --host HOST     Bind host for single-service uvicorn mode (default: $VISUAL_GROUNDING_VIEWER_HOST)
    --port PORT     Bind port for single-service uvicorn mode (default: $VISUAL_GROUNDING_VIEWER_PORT)
    --skip-build    Skip frontend build step in single-service mode and use existing frontend/dist
    --help          Show this help message

Examples:
    $0
    $0 --dev
    $0 --host 127.0.0.1 --port 8004
    $0 --skip-build
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --host)
            VISUAL_GROUNDING_VIEWER_HOST="$2"
            shift 2
            ;;
        --port)
            VISUAL_GROUNDING_VIEWER_PORT="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or not on PATH." >&2
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm is not installed or not on PATH." >&2
    exit 1
fi

if [[ "$DEV_MODE" == true ]]; then
    backend_pid=""

    cleanup() {
        local exit_code=$?
        trap - EXIT INT TERM

        if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" >/dev/null 2>&1; then
            echo
            echo "Stopping backend..."
            kill "$backend_pid" >/dev/null 2>&1 || true
            wait "$backend_pid" 2>/dev/null || true
        fi

        exit "$exit_code"
    }

    trap cleanup EXIT INT TERM

    echo "Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    npm ci

    echo "Syncing Python dependencies..."
    cd "$APP_DIR"
    uv sync

    echo "Starting backend on http://$VISUAL_GROUNDING_VIEWER_DEV_BACKEND_HOST:$VISUAL_GROUNDING_VIEWER_DEV_BACKEND_PORT"
    uv run uvicorn backend.app:app \
        --host "$VISUAL_GROUNDING_VIEWER_DEV_BACKEND_HOST" \
        --port "$VISUAL_GROUNDING_VIEWER_DEV_BACKEND_PORT" &
    backend_pid=$!

    echo "Starting frontend on http://$VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_HOST:$VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_PORT"
    echo "Use Ctrl+C to stop both processes."
    cd "$FRONTEND_DIR"
    npm run dev -- --host "$VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_HOST" --port "$VISUAL_GROUNDING_VIEWER_DEV_FRONTEND_PORT"
    exit 0
fi

if [[ "$SKIP_BUILD" == false ]]; then
    echo "Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    npm ci

    echo "Building frontend..."
    npm run build
    cd "$APP_DIR"
else
    echo "Skipping frontend build."
fi

if [[ ! -f "$FRONTEND_DIR/dist/index.html" ]]; then
    echo "Error: frontend/dist/index.html not found." >&2
    echo "Run without --skip-build to generate the production frontend bundle." >&2
    exit 1
fi

echo "Syncing Python dependencies..."
cd "$APP_DIR"
uv sync

echo "Starting visual grounding viewer on http://$VISUAL_GROUNDING_VIEWER_HOST:$VISUAL_GROUNDING_VIEWER_PORT"
exec uv run uvicorn app:app --host "$VISUAL_GROUNDING_VIEWER_HOST" --port "$VISUAL_GROUNDING_VIEWER_PORT"
