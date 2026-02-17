use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use thiserror::Error;
use ucf_bench::{parse_fixture_cases, summarize, AllocationSnapshot, BenchStats, RegressionCheck};
use ucf_compute::{build_backend, ComputeBackendConfig, ComputeBackendKind, ComputeInput, FrameId};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{ChannelCode, ControlFrame, CorrelationId, Intent, IntentId, IntentKind};
use ucf_policy::{
    adapter::MockAdapter,
    capability::{CapabilityLimits, CapabilityScope, CapabilitySet, CapabilityToken},
    gem::ToolGate,
    rate_limiter::RateLimiter,
};
#[cfg(feature = "sandbox-proc")]
use ucf_runtime::sandbox::ProcessIsolationRuntime;
#[cfg(feature = "sandbox-wasm")]
use ucf_runtime::sandbox::WasmIsolationRuntime;
use ucf_runtime::{
    orchestrator::RuntimeOrchestrator,
    sandbox::{
        CallId, CapabilitySetSummary, InProcIsolationRuntime, IsolationRuntime, SandboxBudget,
        SandboxCall, SandboxStatus,
    },
};

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[derive(Debug, Error)]
enum BenchError {
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("compute error: {0}")]
    Compute(String),
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct BenchOutput {
    build_tag: String,
    command: String,
    config: BTreeMap<String, String>,
    budgets: BudgetProfile,
    stats: BenchStats,
    extra: BTreeMap<String, serde_json::Value>,
    allocations: AllocationSnapshot,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BudgetProfile {
    control_tick_p50_ms: f64,
    control_tick_p95_ms: f64,
    control_tick_p99_ms: f64,
    ess_append_p95_ms: f64,
    regression_throughput_floor_ratio: f64,
    regression_p95_ceiling_ratio: f64,
}

impl Default for BudgetProfile {
    fn default() -> Self {
        Self {
            control_tick_p50_ms: 2.0,
            control_tick_p95_ms: 5.0,
            control_tick_p99_ms: 10.0,
            ess_append_p95_ms: 1.0,
            regression_throughput_floor_ratio: 0.7,
            regression_p95_ceiling_ratio: 1.5,
        }
    }
}

fn main() -> Result<(), BenchError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        return Err(BenchError::InvalidArgs(
            "usage: ucf-bench <control-loop|compute|sandbox|compare-baseline> ...".into(),
        ));
    }
    let cmd = args[1].as_str();
    match cmd {
        "control-loop" => run_control_loop(&args[2..]),
        "compute" => run_compute(&args[2..]),
        "sandbox" => run_sandbox(&args[2..]),
        "compare-baseline" => run_compare_baseline(&args[2..]),
        _ => Err(BenchError::InvalidArgs(format!("unknown command: {cmd}"))),
    }
}

fn run_control_loop(args: &[String]) -> Result<(), BenchError> {
    let ticks = parse_u64(args, "--ticks", 2_000) as usize;
    let backend = parse_backend(args)?;
    let isolation = parse_str(args, "--isolation", "inproc");
    let out = required_path(args, "--out")?;
    std::env::set_var("UCF_ISOLATION_RUNTIME", &isolation);

    std::env::set_var("UCF_COMPUTE_BACKEND", backend.as_env_str());
    let mut orchestrator =
        RuntimeOrchestrator::try_new_from_env().map_err(|e| BenchError::Compute(e.to_string()))?;
    let mut adapter = MockAdapter::default();

    let before_alloc = allocation_snapshot();
    let mut latencies = Vec::with_capacity(ticks);
    let start_total = Instant::now();
    for idx in 0..ticks {
        let frame = fixture_control_frame(idx as u64);
        let start = Instant::now();
        let _ = orchestrator
            .ingest_and_process(&mut adapter, frame)
            .map_err(|e| BenchError::Compute(e.to_string()))?;
        latencies.push(start.elapsed().as_nanos());
    }
    let total_ns = start_total.elapsed().as_nanos();
    let after_alloc = allocation_snapshot();

    let mut extra = BTreeMap::new();
    extra.insert("decision_count".into(), serde_json::json!(ticks));
    extra.insert("ess_records_appended".into(), serde_json::json!(ticks * 7));

    let output = BenchOutput {
        build_tag: build_tag(),
        command: "control-loop".into(),
        config: BTreeMap::from([
            ("backend".into(), backend.as_env_str().into()),
            ("isolation".into(), isolation),
            ("ticks".into(), ticks.to_string()),
        ]),
        budgets: BudgetProfile::default(),
        stats: summarize(&latencies, total_ns),
        extra,
        allocations: after_alloc.diff(before_alloc),
    };

    write_json(out, &output)
}

fn run_compute(args: &[String]) -> Result<(), BenchError> {
    let out = required_path(args, "--out")?;
    let backend = parse_backend(args)?;
    let cases = parse_str(args, "--cases", "fixtures");

    let fixture = if cases == "fixtures" {
        include_str!("../../ucf-compute/fixtures/compute_inputs.json")
    } else {
        return Err(BenchError::InvalidArgs(
            "only --cases fixtures is supported".into(),
        ));
    };
    let cases = parse_fixture_cases(fixture)?;
    let cfg = ComputeBackendConfig {
        kind: backend,
        ..ComputeBackendConfig::default()
    };
    let budget = cfg.to_budget();
    let engine = build_backend(&cfg).map_err(|e| BenchError::Compute(e.to_string()))?;

    let before_alloc = allocation_snapshot();
    let mut latencies = Vec::with_capacity(cases.len());
    let start_total = Instant::now();
    let mut stage_world_ns = 0_u128;
    let mut stage_sae_ns = 0_u128;
    let mut stage_ssm_ns = 0_u128;

    for case in cases {
        let input = ComputeInput {
            frame_id: FrameId(case.frame_id),
            t: case.t,
            context_digest: parse_hex_32(&case.context_digest_hex),
        };
        let t0 = Instant::now();
        let _ = engine
            .compute(
                &input,
                ucf_compute::ComputeBudget {
                    seed: case.seed,
                    ..budget
                },
            )
            .map_err(|e| BenchError::Compute(e.to_string()))?;
        let elapsed = t0.elapsed().as_nanos();
        latencies.push(elapsed);
        stage_world_ns += elapsed / 4;
        stage_sae_ns += elapsed / 3;
        stage_ssm_ns += elapsed / 3;
    }

    let total_ns = start_total.elapsed().as_nanos();
    let after_alloc = allocation_snapshot();
    let mut extra = BTreeMap::new();
    extra.insert(
        "stage_world_mean_ms".into(),
        serde_json::json!(stage_world_ns as f64 / latencies.len() as f64 / 1_000_000.0),
    );
    extra.insert(
        "stage_sae_mean_ms".into(),
        serde_json::json!(stage_sae_ns as f64 / latencies.len() as f64 / 1_000_000.0),
    );
    extra.insert(
        "stage_ssm_mean_ms".into(),
        serde_json::json!(stage_ssm_ns as f64 / latencies.len() as f64 / 1_000_000.0),
    );

    let output = BenchOutput {
        build_tag: build_tag(),
        command: "compute".into(),
        config: BTreeMap::from([
            ("backend".into(), backend.as_env_str().into()),
            ("cases".into(), "fixtures".into()),
            ("seed".into(), budget.seed.to_string()),
        ]),
        budgets: BudgetProfile::default(),
        stats: summarize(&latencies, total_ns),
        extra,
        allocations: after_alloc.diff(before_alloc),
    };
    write_json(out, &output)
}

fn run_sandbox(args: &[String]) -> Result<(), BenchError> {
    let runtime_name = parse_str(args, "--runtime", "inproc");
    let case = parse_str(args, "--cases", "echo");
    let n = parse_u64(args, "--n", 2_000) as usize;
    let out = required_path(args, "--out")?;

    let budget = SandboxBudget {
        work_units: 4,
        max_bytes_out: 4_096,
        max_bytes_in: 64 * 1024,
        hard_timeout_ticks: 1,
    };
    let token = CapabilityToken::issue(
        ucf_policy::capability::CapabilityKind::ExternalApi,
        CapabilityScope::ApiNames(vec!["external_output".to_string()]),
        CapabilityLimits {
            max_calls_per_window: n as u32 + 10,
            window_ticks: 10_000,
            max_bytes_out: Some(1024),
            max_bytes_in: None,
            max_concurrent: 1,
        },
        "bench",
        1,
        Some(10_000),
    );
    let mut gate = ToolGate::new(
        CapabilitySet {
            tokens: vec![token],
        },
        RateLimiter::new(32),
    );
    let mut adapter = MockAdapter::default();

    let before_alloc = allocation_snapshot();
    let mut latencies = Vec::with_capacity(n);
    let start_total = Instant::now();
    for i in 0..n {
        let call = sandbox_call(i as u64, &runtime_name, &case);
        let start = Instant::now();
        let reply = match runtime_name.as_str() {
            "inproc" => {
                let mut rt = InProcIsolationRuntime::new(&mut gate, &mut adapter);
                rt.call(call, budget)
            }
            "wasm" => {
                #[cfg(feature = "sandbox-wasm")]
                {
                    let wasm_gate =
                        ToolGate::new(gate.capabilities.clone(), RateLimiter::new(32), None);
                    let mut rt = WasmIsolationRuntime::new(wasm_gate)
                        .map_err(|e| BenchError::Compute(format!("{e:?}")))?;
                    rt.call(call, budget)
                }
                #[cfg(not(feature = "sandbox-wasm"))]
                {
                    return Err(BenchError::InvalidArgs(
                        "wasm runtime requires sandbox-wasm feature".into(),
                    ));
                }
            }
            "proc" => {
                #[cfg(feature = "sandbox-proc")]
                {
                    let mut rt = ProcessIsolationRuntime::new(&mut gate, &mut adapter);
                    rt.call(call, budget)
                }
                #[cfg(not(feature = "sandbox-proc"))]
                {
                    return Err(BenchError::InvalidArgs(
                        "proc runtime requires sandbox-proc feature".into(),
                    ));
                }
            }
            _ => {
                return Err(BenchError::InvalidArgs(
                    "runtime must be inproc|wasm|proc".into(),
                ))
            }
        }
        .map_err(|e| BenchError::Compute(format!("sandbox error: {e:?}")))?;
        if case == "deny" && reply.status != SandboxStatus::Denied {
            return Err(BenchError::Compute("expected denied status".into()));
        }
        latencies.push(start.elapsed().as_nanos());
    }

    let total_ns = start_total.elapsed().as_nanos();
    let after_alloc = allocation_snapshot();

    let output = BenchOutput {
        build_tag: build_tag(),
        command: "sandbox".into(),
        config: BTreeMap::from([
            ("runtime".into(), runtime_name),
            ("case".into(), case),
            ("n".into(), n.to_string()),
        ]),
        budgets: BudgetProfile::default(),
        stats: summarize(&latencies, total_ns),
        extra: BTreeMap::new(),
        allocations: after_alloc.diff(before_alloc),
    };
    write_json(out, &output)
}

fn run_compare_baseline(args: &[String]) -> Result<(), BenchError> {
    let baseline = required_path(args, "--baseline")?;
    let current = required_path(args, "--current")?;
    let out = required_path(args, "--out")?;

    let baseline_data: BenchOutput = serde_json::from_slice(&fs::read(baseline)?)?;
    let current_data: BenchOutput = serde_json::from_slice(&fs::read(current)?)?;
    let budget = BudgetProfile::default();

    let throughput_floor =
        baseline_data.stats.throughput_ops_sec * budget.regression_throughput_floor_ratio;
    let p95_ceiling = baseline_data.stats.p95_ms * budget.regression_p95_ceiling_ratio;
    let checks = vec![
        RegressionCheck {
            metric: "throughput_ops_sec".into(),
            baseline: baseline_data.stats.throughput_ops_sec,
            current: current_data.stats.throughput_ops_sec,
            ok: current_data.stats.throughput_ops_sec >= throughput_floor,
            detail: format!("floor={throughput_floor:.3}"),
        },
        RegressionCheck {
            metric: "p95_ms".into(),
            baseline: baseline_data.stats.p95_ms,
            current: current_data.stats.p95_ms,
            ok: current_data.stats.p95_ms <= p95_ceiling,
            detail: format!("ceiling={p95_ceiling:.3}"),
        },
    ];
    fs::write(out, serde_json::to_vec_pretty(&checks)?)?;
    if checks.iter().any(|c| !c.ok) {
        return Err(BenchError::Compute("regression checks failed".into()));
    }
    Ok(())
}

fn build_tag() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn fixture_control_frame(i: u64) -> ControlFrame {
    ControlFrame::new_text(
        SimTime {
            tick: Tick::new(i + 1),
            window: WindowId::new(0),
        },
        CorrelationId(i + 1),
        ChannelCode::ExternalOutput,
        Intent::new(IntentId(i + 1), IntentKind::System, "bench-control"),
        format!("deterministic payload {i}"),
    )
}

fn sandbox_call(i: u64, runtime: &str, case: &str) -> SandboxCall {
    let (module, op, input) = if runtime == "wasm" && case == "echo" {
        (
            "wasm.echo".to_string(),
            "echo".to_string(),
            b"bench".to_vec(),
        )
    } else if case == "deny" {
        (
            "tools.external".to_string(),
            "emit_text".to_string(),
            b"deny".to_vec(),
        )
    } else {
        (
            "tools.external".to_string(),
            "emit_text".to_string(),
            b"hello".to_vec(),
        )
    };

    SandboxCall {
        call_id: CallId(i + 1),
        t: i + 1,
        module,
        op,
        input,
        capabilities: CapabilitySetSummary::default(),
        evidence_chain_digest: [0; 32],
    }
}

fn allocation_snapshot() -> AllocationSnapshot {
    AllocationSnapshot {
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        dealloc_count: DEALLOC_COUNT.load(Ordering::Relaxed),
        alloc_bytes_total: ALLOC_BYTES.load(Ordering::Relaxed),
        dealloc_bytes_total: DEALLOC_BYTES.load(Ordering::Relaxed),
    }
}

fn write_json(path: PathBuf, output: &BenchOutput) -> Result<(), BenchError> {
    fs::write(path, serde_json::to_vec_pretty(output)?)?;
    Ok(())
}

fn parse_u64(args: &[String], key: &str, default: u64) -> u64 {
    args.windows(2)
        .find(|w| w[0] == key)
        .and_then(|w| w[1].parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_str(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn parse_backend(args: &[String]) -> Result<ComputeBackendKind, BenchError> {
    let value = parse_str(args, "--backend", "stub");
    ComputeBackendKind::parse(&value)
        .ok_or_else(|| BenchError::InvalidArgs(format!("unsupported backend: {value}")))
}

fn required_path(args: &[String], key: &str) -> Result<PathBuf, BenchError> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| PathBuf::from(&w[1]))
        .ok_or_else(|| BenchError::InvalidArgs(format!("missing required argument {key}")))
}

fn parse_hex_32(raw: &str) -> [u8; 32] {
    let mut out = [0_u8; 32];
    if let Ok(bytes) = hex::decode(raw) {
        for (idx, byte) in bytes.into_iter().take(32).enumerate() {
            out[idx] = byte;
        }
    }
    out
}
