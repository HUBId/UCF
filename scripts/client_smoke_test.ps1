param(
  [string]$OutDir = "./out/client_smoke"
)

$ErrorActionPreference = "Stop"
$Endpoint = if ($env:UCF_CLIENT_ENDPOINT) { $env:UCF_CLIENT_ENDPOINT } else { "tcp://127.0.0.1:44991" }
$Auth = if ($env:UCF_CLIENT_AUTH) { $env:UCF_CLIENT_AUTH } else { "test-token" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
"endpoint=$Endpoint" | Out-File -FilePath "$OutDir/meta.txt" -Encoding utf8
"auth_set=$([string]::IsNullOrEmpty($Auth) -eq $false)" | Out-File -FilePath "$OutDir/meta.txt" -Append -Encoding utf8

cargo run -p ucf-client -- --endpoint $Endpoint --auth $Auth submit --fixture fixtures/client/controlframe_min.json | Tee-Object -FilePath "$OutDir/01_submit.txt"
cargo run -p ucf-client -- --endpoint $Endpoint --auth $Auth stream --max 1 | Tee-Object -FilePath "$OutDir/02_stream.txt"
cargo run -p ucf-client -- --endpoint $Endpoint --auth $Auth ess query --last 32 | Tee-Object -FilePath "$OutDir/03_ess.txt"
cargo run -p ucf-client -- --endpoint $Endpoint --auth $Auth report explain-tick --t 1 | Tee-Object -FilePath "$OutDir/04_explain_tick.txt"
cargo run -p ucf-client -- --endpoint $Endpoint --auth $Auth report readiness-gate --latest | Tee-Object -FilePath "$OutDir/05_readiness_gate.txt"

"client_smoke_out=$OutDir"
