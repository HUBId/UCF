#!/usr/bin/env bash
set -euo pipefail

SOAK_MINUTES="${SOAK_MINUTES:-30}"
OUT_ROOT="${1:-./out/rc1}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_ROOT}/${TS}/soak"
mkdir -p "${OUT_DIR}"

export UCF_OFFLINE=1
export UCF_TOOLS_DEFAULT=deny
export UCF_PROFILE=test

TICKS="$((SOAK_MINUTES * 60))"

cargo run -p ucf-ops -- bench \
  --scenario fixtures/e2e_scenario_a.json \
  --ticks "${TICKS}" \
  --rss-cap-mb 3072 \
  --out "${OUT_DIR}/scenario_a_soak_report.json"

cargo run -p ucf-ops -- bench \
  --scenario fixtures/e2e_scenario_ebm_v1.json \
  --ticks "${TICKS}" \
  --rss-cap-mb 3072 \
  --out "${OUT_DIR}/scenario_ebm_soak_report.json"

cargo run -p ucf-ops -- readiness-gate --profile test --out "${OUT_DIR}/readiness_gate.json"
cargo run -p ucf-ops -- out manifest --dir "${OUT_DIR}" > "${OUT_DIR}/manifest_pretty.json"
echo "rc1_soak_out=${OUT_DIR}"
