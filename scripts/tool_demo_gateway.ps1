param(
  [string]$OutDir = "./out/tool_demo"
)

$ErrorActionPreference = "Stop"
$workdir = if ($env:UCF_WORKDIR) { $env:UCF_WORKDIR } else { ".ucf" }
$endpoint = if ($env:UCF_CLIENT_ENDPOINT) { $env:UCF_CLIENT_ENDPOINT } else { "tcp://127.0.0.1:44991" }
$auth = if ($env:UCF_CLIENT_AUTH) { $env:UCF_CLIENT_AUTH } else { "test-token" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$env:UCF_POLICY_OVERLAY = "demo_toolread"
$env:UCF_GATEWAY_TOKEN = if ($env:UCF_GATEWAY_TOKEN) { $env:UCF_GATEWAY_TOKEN } else { $auth }
$env:UCF_CLIENT_AUTH = $auth

cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_tool_plan_demo.json --ticks 8 --out ./out | Tee-Object -FilePath "$OutDir/01_bringup.txt"
$gateway = Start-Process cargo -ArgumentList "run","-p","ucf-gateway","--bin","ucf-gateway-local" -NoNewWindow -PassThru -RedirectStandardOutput "$OutDir/gateway.log" -RedirectStandardError "$OutDir/gateway.err.log"
Start-Sleep -Seconds 1

try {
  cargo run -p ucf-client -- --endpoint $endpoint --auth $auth submit --fixture fixtures/client/controlframe_tool_demo.json | Tee-Object -FilePath "$OutDir/03_submit.txt"
  cargo run -p ucf-client -- --endpoint $endpoint --auth $auth stream --max 1 | Tee-Object -FilePath "$OutDir/04_stream.txt"
  cargo run -p ucf-client -- --endpoint $endpoint --auth $auth ess query --last 64 | Tee-Object -FilePath "$OutDir/04_ess.txt"
  cargo run -p ucf-client -- --endpoint $endpoint --auth $auth report explain-tick --t 10 | Tee-Object -FilePath "$OutDir/05_explain_tick.txt"
  cargo run -p ucf-ops -- replay --from 0 --to 999999 --report "$OutDir/replay_verify_report.json" | Tee-Object -FilePath "$OutDir/06_replay_verify.txt"
}
finally {
  if ($gateway -and !$gateway.HasExited) {
    Stop-Process -Id $gateway.Id -Force
  }
}

"tool_demo_out=$OutDir"
