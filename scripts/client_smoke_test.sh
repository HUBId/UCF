#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./out/client_smoke}"
ENDPOINT="${UCF_CLIENT_ENDPOINT:-unix:///tmp/ucf_gateway_v1.sock}"
AUTH="${UCF_CLIENT_AUTH:-test-token}"

mkdir -p "${OUT_DIR}"

echo "endpoint=${ENDPOINT}" > "${OUT_DIR}/meta.txt"
echo "auth_set=$([ -n "${AUTH}" ] && echo yes || echo no)" >> "${OUT_DIR}/meta.txt"

echo "[1/5] submit fixture"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" submit \
  --fixture fixtures/client/controlframe_min.json | tee "${OUT_DIR}/01_submit.txt"

echo "[2/5] stream decisions"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" stream --max 1 \
  | tee "${OUT_DIR}/02_stream.txt"

echo "[3/5] query ess"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" ess query --last 32 \
  | tee "${OUT_DIR}/03_ess.txt"

echo "[4/5] report explain-tick"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" report explain-tick --t 1 \
  | tee "${OUT_DIR}/04_explain_tick.txt"

echo "[5/5] report readiness-gate"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" report readiness-gate --latest \
  | tee "${OUT_DIR}/05_readiness_gate.txt"

echo "client_smoke_out=${OUT_DIR}"
