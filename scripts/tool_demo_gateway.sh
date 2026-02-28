#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./out/tool_demo}"
WORKDIR="${UCF_WORKDIR:-.ucf}"
ENDPOINT="${UCF_CLIENT_ENDPOINT:-unix:///tmp/ucf_gateway_v1.sock}"
AUTH="${UCF_CLIENT_AUTH:-test-token}"

mkdir -p "${OUT_DIR}"
export UCF_POLICY_OVERLAY="demo_toolread"
export UCF_GATEWAY_TOKEN="${UCF_GATEWAY_TOKEN:-$AUTH}"
export UCF_CLIENT_AUTH="$AUTH"

echo "[1/6] bringup in demo overlay"
cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_tool_plan_demo.json --ticks 8 --out ./out \
  | tee "${OUT_DIR}/01_bringup.txt"

echo "[2/6] start local gateway"
cargo run -p ucf-gateway --bin ucf-gateway-local > "${OUT_DIR}/gateway.log" 2>&1 &
GATEWAY_PID=$!
trap 'kill ${GATEWAY_PID} >/dev/null 2>&1 || true' EXIT
sleep 1

echo "[3/6] submit demo control frame"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" submit \
  --fixture fixtures/client/controlframe_tool_demo.json | tee "${OUT_DIR}/03_submit.txt"

echo "[4/6] stream one decision + query ess"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" stream --max 1 \
  | tee "${OUT_DIR}/04_stream.txt"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" ess query --last 64 \
  | tee "${OUT_DIR}/04_ess.txt"

echo "[5/6] explain tick for tool execution window"
cargo run -p ucf-client -- --endpoint "${ENDPOINT}" --auth "${AUTH}" report explain-tick --t 10 \
  | tee "${OUT_DIR}/05_explain_tick.txt"

echo "[6/6] replay verify-only"
cargo run -p ucf-ops -- replay --from 0 --to 999999 --report "${OUT_DIR}/replay_verify_report.json" \
  | tee "${OUT_DIR}/06_replay_verify.txt"

echo "tool_demo_out=${OUT_DIR}"
