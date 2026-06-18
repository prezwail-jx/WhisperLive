#!/usr/bin/env bash
set -e

MODE="$1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGINX_DIR="$PROJECT_ROOT/deploy/nginx"

SINGLE_CONF="$NGINX_DIR/whisperlive-single.conf"
DUAL_CONF="$NGINX_DIR/whisperlive-dual.conf"
ACTIVE_CONF="$NGINX_DIR/whisperlive.conf"

if [ "$MODE" = "single" ]; then
    cp "$SINGLE_CONF" "$ACTIVE_CONF"
    echo "[OK] Switched Nginx config to SINGLE GPU mode."
elif [ "$MODE" = "dual" ]; then
    cp "$DUAL_CONF" "$ACTIVE_CONF"
    echo "[OK] Switched Nginx config to DUAL GPU mode."
else
    echo "Usage:"
    echo "  $0 single"
    echo "  $0 dual"
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -q '^whisperlive-web-gateway$'; then
    docker restart whisperlive-web-gateway >/dev/null
    echo "[OK] Restarted whisperlive-web-gateway."
else
    echo "[INFO] whisperlive-web-gateway is not running. Start it manually when ready."
fi
