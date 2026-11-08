#!/usr/bin/env bash
# Usage: deploy/deploy.sh <app folder> [branch]
#   HEALTH_URL=https://geoai.example.com/health deploy/deploy.sh /path/to/argo_geoai
set -euo pipefail

UV="${UV:-$HOME/.local/bin/uv}"
HEALTH_URL="${HEALTH_URL:-}"

usage() {
    echo "usage: $0 <app folder> [branch]" >&2
    exit 2
}

check_app_root() {
    local folder="$1"
    if [[ ! -d "$folder" ]]; then
        echo "error: folder does not exist: $folder" >&2
        exit 1
    fi
    if [[ ! -d "$folder/.git" ]]; then
        echo "error: not a git checkout: $folder" >&2
        exit 1
    fi
    if [[ ! -f "$folder/pyproject.toml" ]] || ! grep -q '^name = "argo-geoai"' "$folder/pyproject.toml"; then
        echo "error: not an argo_geoai checkout: $folder" >&2
        exit 1
    fi
}

[[ $# -ge 1 && $# -le 2 ]] || usage
APP_ROOT="$1"
BRANCH="${2:-main}"
check_app_root "$APP_ROOT"

cd "$APP_ROOT"
echo "== code: $BRANCH"
git fetch --prune origin
git checkout --quiet "$BRANCH"
git pull --ff-only origin "$BRANCH"
git log -1 --format='%h %s (%ci)'

echo "== dependencies (exactly uv.lock, without dev tools)"
"$UV" sync --frozen --no-dev

echo "== the code imports and sees its tools"
.venv/bin/argo-geoai version
.venv/bin/argo-geoai tools

echo "== restart"
mkdir -p tmp
touch tmp/restart.txt

if [[ -n "$HEALTH_URL" ]]; then
    for _ in $(seq 1 30); do
        if curl -fsS "$HEALTH_URL"; then
            echo
            echo "== up"
            exit 0
        fi
        sleep 2
    done
    echo "health check failed: $HEALTH_URL (the app's output: /var/log/nginx/error.log)" >&2
    exit 1
fi
