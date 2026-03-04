# Zero-Trust Local v1

## Ziel

`localhost` ist keine vertrauenswürdige Sicherheitsgrenze. Gateway und Client werden daher standardmäßig auf lokale IPC-Transporte gehärtet.

## Empfohlene Transporte je OS

- **Linux/macOS (Standard):** Unix Domain Socket unter
  - `.ucf/data/ipc/gateway.sock`
- **Windows (Standard):** Named Pipe
  - `\\.\pipe\ucf_gateway`
- **TCP (nur explizit/dev):**
  - `UCF_GATEWAY_TRANSPORT=tcp`
  - und zwingend `UCF_GATEWAY_BIND=127.0.0.1:<port>`

## Gateway starten

### Linux/macOS (Unix Socket Standard)

```bash
UCF_GATEWAY_TOKEN=test-token cargo run -p ucf-gateway --bin ucf-gateway-local
```

Optional explizit:

```bash
UCF_GATEWAY_TRANSPORT=unix UCF_GATEWAY_TOKEN=test-token cargo run -p ucf-gateway --bin ucf-gateway-local
```

### Windows (Named Pipe Standard)

```powershell
$env:UCF_GATEWAY_TOKEN="test-token"
cargo run -p ucf-gateway --bin ucf-gateway-local
```

Optional explizit:

```powershell
$env:UCF_GATEWAY_TRANSPORT="pipe"
$env:UCF_GATEWAY_TOKEN="test-token"
cargo run -p ucf-gateway --bin ucf-gateway-local
```

### TCP nur explizit und loopback-only

```bash
UCF_GATEWAY_TRANSPORT=tcp UCF_GATEWAY_BIND=127.0.0.1:44991 UCF_GATEWAY_TOKEN=test-token cargo run -p ucf-gateway --bin ucf-gateway-local
```

## Client Endpunkte

- Unix: `--endpoint unix://.ucf/data/ipc/gateway.sock`
- Pipe: `--endpoint pipe://./pipe/ucf_gateway`
- TCP: `--endpoint tcp://127.0.0.1:44991`

## Health Probe (lokal, sicher)

Gateway-Health ist ein lokaler IPC-Endpunkt (`health`) mit stabiler, begrenzter Oberfläche.

- `test`/`prod`: Token erforderlich (`health:read`)
- `dev`: leerer Token für Health erlaubt, aber mit Warnung

Beispiel:

```bash
cargo run -p ucf-ops -- health check --endpoint unix://.ucf/data/ipc/gateway.sock --auth test-token --out ./out/health.json
```

## Lokale Firewall Guidance (ohne Tools)

### Windows Defender Firewall

1. Eingehende Regel für Gateway-Binary prüfen.
2. Netzwerkprofil auf **Privat** oder **Domäne** beschränken.
3. Scope auf **Local IP only** (keine Remote-Adressen) setzen.
4. Falls TCP genutzt wird: explizit nur `127.0.0.1` erlauben.

### Linux (ufw)

1. Falls nötig, deny-by-default für eingehend aktivieren.
2. Keine generelle Freigabe für Gateway-Port setzen.
3. Wenn TCP dev-only benötigt wird: Zugriff auf `127.0.0.1` begrenzen.

## Threat Harness ausführen

```bash
cargo run -p ucf-ops -- gateway threat-test --out ./out/gateway_threat.json
```

Harness beinhaltet deterministisch und begrenzt:

1. **Auth brute** (20 invalide Tokens) → `ERR_AUTH_DENIED`
2. **Flood** (200 Requests Burst) → `ERR_RATE_LIMITED`
3. **Malformed Frames** (bounded random bytes) → `ERR_SCHEMA_INVALID`
4. **Version Mismatch** (`schema_version` unsupported) → `ERR_VERSION_MISMATCH`

## Report interpretieren

Datei: `./out/gateway_threat.json`

- `ok=true` bedeutet alle vier Cases erfüllt.
- `cases[*].observed_error_count` sollte den Erwartungen entsprechen.
- `abuse_log_total` zeigt persistierte Abuse-Records im Lauf.
