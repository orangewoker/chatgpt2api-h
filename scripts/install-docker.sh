#!/usr/bin/env bash
set -Eeuo pipefail

# ChatGPT2API Docker installer for fnOS and 1Panel.
# Usage:
#   bash install-docker.sh
#   INSTALL_DIR=/opt/chatgpt2api CHATGPT2API_PORT=8080 bash install-docker.sh

IMAGE="${CHATGPT2API_IMAGE:-ghcr.io/basketikun/chatgpt2api:latest}"
INSTALL_DIR="${INSTALL_DIR:-/opt/chatgpt2api}"
RAW_BASE_URL="${RAW_BASE_URL:-https://raw.githubusercontent.com/orangewoker/chatgpt2api-h/main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { printf '\033[1;32m[chatgpt2api]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[chatgpt2api]\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "未检测到 Docker，请先在飞牛 OS 或 1Panel 安装 Docker。"
docker compose version >/dev/null 2>&1 || die "未检测到 Docker Compose v2，请升级 Docker。"

mkdir -p "$INSTALL_DIR/data"

copy_or_download() {
  local local_path="$1" remote_name="$2"
  if [[ -f "$local_path" ]]; then
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$RAW_BASE_URL/$remote_name" -o "$local_path" || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$local_path" "$RAW_BASE_URL/$remote_name" || return 1
  else
    return 1
  fi
}

if [[ -f "$PROJECT_DIR/docker-compose.deploy.yml" ]]; then
  cp "$PROJECT_DIR/docker-compose.deploy.yml" "$INSTALL_DIR/docker-compose.yml"
else
  copy_or_download "$INSTALL_DIR/docker-compose.yml" "docker-compose.deploy.yml" \
    || die "无法下载 docker-compose.deploy.yml，请把该文件上传到安装目录后重试。"
fi

if [[ ! -f "$INSTALL_DIR/config.json" ]]; then
  if [[ -f "$PROJECT_DIR/config.json" ]]; then
    cp "$PROJECT_DIR/config.json" "$INSTALL_DIR/config.json"
  elif ! copy_or_download "$INSTALL_DIR/config.json" "config.json"; then
    printf '{}\n' > "$INSTALL_DIR/config.json"
    log "未下载到 config.json，已创建空配置，应用将使用内置默认值。"
  fi
fi

if [[ ! -f "$INSTALL_DIR/.env" ]]; then
  cat > "$INSTALL_DIR/.env" <<EOF
CHATGPT2API_IMAGE=$IMAGE
CHATGPT2API_PORT=${CHATGPT2API_PORT:-3000}
CHATGPT2API_CONTAINER_NAME=${CHATGPT2API_CONTAINER_NAME:-chatgpt2api}
STORAGE_BACKEND=${STORAGE_BACKEND:-sqlite}
DATABASE_URL=${DATABASE_URL:-sqlite:////app/data/accounts.db}
CHATGPT2API_AUTH_KEY=${CHATGPT2API_AUTH_KEY:-}
CHATGPT2API_BASE_URL=${CHATGPT2API_BASE_URL:-}
EOF
  chmod 600 "$INSTALL_DIR/.env" 2>/dev/null || true
fi

cd "$INSTALL_DIR"
log "拉取镜像: $IMAGE"
docker compose -f docker-compose.yml pull
log "启动服务"
docker compose -f docker-compose.yml up -d
docker compose -f docker-compose.yml ps

PORT="${CHATGPT2API_PORT:-3000}"
log "安装完成: http://127.0.0.1:$PORT"
log "安装目录: $INSTALL_DIR"
log "查看日志: cd $INSTALL_DIR && docker compose logs -f app"
