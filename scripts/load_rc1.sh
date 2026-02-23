#!/usr/bin/env bash
set -euo pipefail

DURATION_SECONDS="${DURATION_SECONDS:-300}"
OUT_ROOT="${1:-./out/rc1}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_ROOT}/${TS}/load"
mkdir -p "${OUT_DIR}"

export UCF_OFFLINE=1
export UCF_TOOLS_DEFAULT=deny
export UCF_PROFILE=test

run_bench() {
  local scenario="$1"
  local name="$2"
  cargo run -p ucf-ops -- bench \
    --scenario "${scenario}" \
    --ticks "${DURATION_SECONDS}" \
    --rss-cap-mb 2048 \
    --out "${OUT_DIR}/${name}_bench_report.json"
}

run_bench fixtures/e2e_scenario_a.json scenario_a
run_bench fixtures/e2e_scenario_ebm_v1.json scenario_ebm

cargo run -p ucf-ops -- out manifest --dir "${OUT_DIR}" > "${OUT_DIR}/manifest_pretty.json"
echo "rc1_load_out=${OUT_DIR}"
