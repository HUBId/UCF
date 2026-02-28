#!/usr/bin/env sh
set -eu

if [ "$#" -lt 2 ]; then
  echo "usage: upgrade_bundle.sh <upgrade|rollback> <bundle_id> [--health-cmd '<cmd>']"
  exit 1
fi

ACTION="$1"
BUNDLE_ID="$2"
HEALTH_CMD="./bin/ucf-ops health check --bundle . --out ./out/health.json"
if [ "${3:-}" = "--health-cmd" ] && [ -n "${4:-}" ]; then
  HEALTH_CMD="$4"
fi

ROOT="$(pwd)"
BUNDLES_DIR="$ROOT/bundles"
CURRENT_LINK="$BUNDLES_DIR/current"
PREVIOUS_LINK="$BUNDLES_DIR/previous"
TARGET="$BUNDLES_DIR/releases/$BUNDLE_ID"

mkdir -p "$BUNDLES_DIR/releases"
[ -d "$TARGET" ] || { echo "missing target bundle: $TARGET"; exit 2; }

if [ -L "$CURRENT_LINK" ] || [ -e "$CURRENT_LINK" ]; then
  CUR_REAL="$(cd "$(dirname "$CURRENT_LINK")" && cd "$(readlink "$CURRENT_LINK")" && pwd)"
  rm -f "$PREVIOUS_LINK"
  ln -s "$CUR_REAL" "$PREVIOUS_LINK"
fi

case "$ACTION" in
  upgrade)
    rm -f "$CURRENT_LINK"
    ln -s "$TARGET" "$CURRENT_LINK"
    ;;
  rollback)
    [ -L "$PREVIOUS_LINK" ] || { echo "no previous bundle"; exit 3; }
    PREV_REAL="$(cd "$(dirname "$PREVIOUS_LINK")" && cd "$(readlink "$PREVIOUS_LINK")" && pwd)"
    rm -f "$CURRENT_LINK"
    ln -s "$PREV_REAL" "$CURRENT_LINK"
    ;;
  *)
    echo "invalid action: $ACTION"
    exit 1
    ;;
esac

cd "$CURRENT_LINK"
sh -c "$HEALTH_CMD"

echo "status=ok action=$ACTION current=$(pwd)"
