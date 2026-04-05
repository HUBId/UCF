use std::collections::BTreeMap;
use std::fs;

use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::Digest;
use ucf_compute::capabilities::{LlmOutputClass, LlmRequest};
use ucf_compute::feature_extractor::ToySaeExtractor;
use ucf_compute::lfm::LfmInput;
use ucf_compute::ssm::SsmInput;
use ucf_compute::world_model::{StageQuality, WorldModelInput};
use ucf_compute::{
    clamp01, compute_input_from_control, fuse_signals, BackendPackConfig, BackendPackFactory,
    ComputeBackendConfig,
};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{ChannelCode, ControlFrame, CorrelationId, Intent, IntentId, IntentKind};
use ucf_policy::candidate::{
    CandidateGenerator, DecisionBudget, DecisionContext, DefaultCandidateGeneratorV0,
};
use ucf_runtime::ebm::{
    active_ebm_constraints, candidate_feature_from_decision, CpuEbmStubV0, EbmInput, EbmReasoner,
    EbmSignals,
};

use crate::OpsError;

const BENCH_SCHEMA_VERSION: u16 = 1;
const DEFAULT_RSS_SAMPLE_EVERY: u64 = 16;

fn obs_features_from_context(context_digest: [u8; 32]) -> [f32; 16] {
    let mut obs = [0.0_f32; 16];
    for (i, slot) in obs.iter_mut().enumerate() {
        let a = context_digest[i] as f32 / 255.0;
        let b = context_digest[i + 16] as f32 / 255.0;
        *slot = ((0.65 * a + 0.35 * b) * 2.0 - 1.0).clamp(-1.0, 1.0);
    }
    obs
}

#[derive(Debug, Clone)]
pub struct BenchArgs {
    pub scenario: PathBuf,
    pub ticks: u64,
    pub out: PathBuf,
    pub rss_sample_every: u64,
    pub rss_cap_mb: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScenarioFixture {
    scenario: String,
    ticks: usize,
    channel: String,
    intent_summary: String,
    signal_values: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub schema_version: u16,
    pub run_id: String,
    pub scenario_id: String,
    pub scenario_digest: String,
    pub backend_pack_digest: String,
    pub ticks: u64,
    pub throughput_ticks_per_sec: f64,
    pub tick_time_ms: StageLatencyStats,
    pub stage_latency_ms: BTreeMap<String, StageLatencyStats>,
    pub counters: BenchCounters,
    pub memory: MemoryStats,
    pub hints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageLatencyStats {
    pub samples: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchCounters {
    pub degraded_stages: BTreeMap<String, u64>,
    pub budget_exceeded_events: u64,
    pub backpressure_ticks: u64,
    pub queue_depth_avg: f64,
    pub tool_issuance_denies: u64,
    pub tool_issuance_attempts: u64,
    pub ebm_candidates_scored_total: u64,
    pub ebm_mean_k_q: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub mode: String,
    pub samples: usize,
    pub min_rss_mb: Option<f64>,
    pub mean_rss_mb: Option<f64>,
    pub max_rss_mb: Option<f64>,
    pub cap_mb: Option<u64>,
    pub cap_exceeded: bool,
}

#[derive(Debug, Default)]
struct QuantileEstimator {
    values: Vec<f64>,
}

impl QuantileEstimator {
    fn push(&mut self, value_ms: f64) {
        self.values.push(value_ms.max(0.0));
    }

    fn stats(&self) -> StageLatencyStats {
        if self.values.is_empty() {
            return StageLatencyStats {
                samples: 0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                max_ms: 0.0,
            };
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(f64::total_cmp);
        let q = |v: f64| -> f64 {
            let idx = (((sorted.len() - 1) as f64) * v).round() as usize;
            sorted[idx]
        };
        StageLatencyStats {
            samples: sorted.len(),
            p50_ms: q(0.50),
            p95_ms: q(0.95),
            p99_ms: q(0.99),
            max_ms: *sorted.last().unwrap_or(&0.0),
        }
    }
}

fn read_rss_mb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb = rest.split_whitespace().next()?.parse::<f64>().ok()?;
                return Some(kb / 1024.0);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn channel_from_fixture(value: &str) -> Result<ChannelCode, OpsError> {
    match value {
        "external_output" => Ok(ChannelCode::ExternalOutput),
        "internal_thought" => Ok(ChannelCode::InternalThought),
        other => Err(OpsError::Invalid(format!(
            "unsupported channel fixture value: {other}"
        ))),
    }
}

pub fn bench_run(args: &BenchArgs) -> Result<BenchReport, OpsError> {
    std::env::set_var("UCF_OFFLINE", "1");
    std::env::set_var("UCF_TOOLS_DEFAULT", "deny");

    if std::env::var("UCF_POLICY_BUNDLE_SHA256").is_err() {
        let manifest_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/manifest.toml");
        if let Ok(manifest_raw) = fs::read_to_string(manifest_path) {
            if let Some(hash) = manifest_raw.lines().find_map(|line| {
                line.trim()
                    .strip_prefix("bundle_sha256 = ")
                    .and_then(|rest| rest.strip_prefix('"'))
                    .and_then(|rest| rest.strip_suffix('"'))
            }) {
                std::env::set_var("UCF_POLICY_BUNDLE_SHA256", hash);
            }
        }
    }

    let fixture_raw = fs::read_to_string(&args.scenario)?;
    let fixture: ScenarioFixture = serde_json::from_str(&fixture_raw)?;
    if fixture.signal_values.is_empty() {
        return Err(OpsError::Invalid(
            "scenario signal_values cannot be empty".to_string(),
        ));
    }
    let scenario_digest = hex::encode(sha2::Sha256::digest(fixture_raw.as_bytes()));

    let pack = BackendPackFactory::build(BackendPackConfig::from_env()?)?;
    let budget = ComputeBackendConfig::from_env()?.to_budget();
    let channel = channel_from_fixture(&fixture.channel)?;
    let tick_cap = args.ticks.min(fixture.ticks as u64).max(1);

    let mut stage_stats: BTreeMap<String, QuantileEstimator> =
        ["world", "sae", "ssm", "lfm", "risk", "governor", "llm"]
            .iter()
            .map(|k| (k.to_string(), QuantileEstimator::default()))
            .collect();
    let mut tick_stats = QuantileEstimator::default();
    let mut rss_samples = Vec::new();
    let mut counters = BenchCounters {
        degraded_stages: BTreeMap::new(),
        budget_exceeded_events: 0,
        backpressure_ticks: 0,
        queue_depth_avg: 0.0,
        tool_issuance_denies: 0,
        tool_issuance_attempts: 0,
        ebm_candidates_scored_total: 0,
        ebm_mean_k_q: 0,
    };
    let mut ebm = CpuEbmStubV0;
    let generator = DefaultCandidateGeneratorV0;
    let _constraints = active_ebm_constraints();

    let run_start = Instant::now();
    let run_id = format!("bench-{}", crate::now_unix_secs());
    for idx in 0..tick_cap {
        let sig = fixture.signal_values[(idx as usize) % fixture.signal_values.len()];
        let tick = idx + 1;
        let corr = 60_000 + tick;
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(corr),
            channel,
            Intent::new(
                IntentId(90),
                IntentKind::System,
                fixture.intent_summary.as_str(),
            ),
            format!("sig:{sig:03}:scenario:{}:tick:{tick:03}", fixture.scenario),
        );
        let input = compute_input_from_control(&ctrl);

        let tick_start = Instant::now();

        // explicit stage timings (best-effort)
        let t0 = Instant::now();
        let world_input = WorldModelInput {
            t: input.t,
            context_digest: input.context_digest,
            previous_state_digest: None,
            obs_features: obs_features_from_context(input.context_digest),
            seed: budget.seed,
        };
        let world_out = pack
            .world()
            .lock()
            .map_err(|_| OpsError::Invalid("world model mutex poisoned".to_string()))?
            .step(&world_input, budget)
            .ok();
        stage_stats
            .get_mut("world")
            .expect("stage exists")
            .push(t0.elapsed().as_secs_f64() * 1000.0);

        let t1 = Instant::now();
        let evidence_seed: [u8; 32] = sha2::Sha256::digest(input.context_digest).into();
        let sae_in = ToySaeExtractor::make_input(
            &input,
            &world_out.clone().unwrap_or_else(|| {
                ucf_compute::world_model::WorldModelOutput::degraded_budget(
                    "bench/world_unavailable",
                )
            }),
            budget.seed,
            evidence_seed,
        );
        let sae_out = pack.sae().extract(&sae_in, budget).ok();
        stage_stats
            .get_mut("sae")
            .expect("stage exists")
            .push(t1.elapsed().as_secs_f64() * 1000.0);

        let t2 = Instant::now();
        let ssm_in = SsmInput {
            t: input.t,
            spikes_digest: sae_out.as_ref().map(|s| s.spikes_digest).unwrap_or([0; 32]),
            spike_count: sae_out.as_ref().map(|s| s.spike_count).unwrap_or(0),
            sae_energy: sae_out.as_ref().map(|s| s.energy).unwrap_or(0.0),
            world_surprise: world_out.as_ref().map(|w| w.surprise).unwrap_or(0.0),
            risk: 0.0,
            seed: budget.seed,
            context_digest: input.context_digest,
        };
        let ssm_out = pack
            .ssm()
            .lock()
            .map_err(|_| OpsError::Invalid("ssm mutex poisoned".to_string()))?
            .step(&ssm_in, budget)
            .ok();
        stage_stats
            .get_mut("ssm")
            .expect("stage exists")
            .push(t2.elapsed().as_secs_f64() * 1000.0);

        let t3 = Instant::now();
        let lfm_in = LfmInput {
            t: input.t,
            context_digest: input.context_digest,
            world_digest: world_out
                .as_ref()
                .map(|w| w.prediction_digest)
                .unwrap_or([0; 32]),
            surprise: world_out.as_ref().map(|w| w.surprise).unwrap_or(0.0),
            spikes_digest: sae_out.as_ref().map(|s| s.spikes_digest).unwrap_or([0; 32]),
            spike_count: sae_out.as_ref().map(|s| s.spike_count).unwrap_or(0),
            sae_energy: sae_out.as_ref().map(|s| s.energy).unwrap_or(0.0),
            pressure: ssm_out.as_ref().map(|s| s.pressure).unwrap_or(0.0),
            coherence: None,
            instability: None,
            hormone_stress: None,
            neuro_arousal: None,
            governor_tier: Some(0),
            prediction_error: world_out.as_ref().map(|w| w.prediction_error),
            risk: None,
            confidence: None,
            prior_uncertainty: None,
            seed: budget.seed,
        };
        let lfm_out = pack
            .lfm()
            .lock()
            .map_err(|_| OpsError::Invalid("lfm mutex poisoned".to_string()))?
            .step(&lfm_in, budget)
            .ok();
        stage_stats
            .get_mut("lfm")
            .expect("stage exists")
            .push(t3.elapsed().as_secs_f64() * 1000.0);

        let t4 = Instant::now();
        let (base_risk, base_confidence) = fuse_signals(
            world_out.as_ref().map(|w| w.surprise).unwrap_or(0.0),
            ssm_out.as_ref().map(|s| s.pressure).unwrap_or(0.0),
            sae_out.as_ref().map(|s| s.energy).unwrap_or(0.0),
        );
        let risk =
            clamp01(base_risk + 0.2 * lfm_out.as_ref().map(|l| l.uncertainty).unwrap_or(0.0));
        let confidence =
            clamp01(base_confidence * lfm_out.as_ref().map(|l| l.stability).unwrap_or(1.0));
        stage_stats
            .get_mut("risk")
            .expect("stage exists")
            .push(t4.elapsed().as_secs_f64() * 1000.0);

        let t5 = Instant::now();
        let backpressure = risk > 0.8 || confidence < 0.2;
        if backpressure {
            counters.backpressure_ticks = counters.backpressure_ticks.saturating_add(1);
        }
        stage_stats
            .get_mut("governor")
            .expect("stage exists")
            .push(t5.elapsed().as_secs_f64() * 1000.0);

        let t_ebm = Instant::now();
        let decision_ctx = DecisionContext {
            now_t: tick,
            risk,
            confidence,
            evidence_chain_digest: input.context_digest,
            planning_allowed: true,
            liquid_context: None,
        };
        let candidates = generator.generate(&ctrl, &decision_ctx, DecisionBudget::default());
        let ebm_input = EbmInput {
            t: tick,
            governor_tier: 0,
            emergency_active: false,
            context_digest: input.context_digest,
            signals: EbmSignals {
                risk_q: ucf_types::UQ0_16::from_f32_clamped(risk),
                confidence_q: ucf_types::UQ0_16::from_f32_clamped(confidence),
                pressure_q: ucf_types::UQ0_16::from_f32_clamped(
                    ssm_out.as_ref().map(|s| s.pressure).unwrap_or(0.0),
                ),
                surprise_q: ucf_types::UQ0_16::from_f32_clamped(
                    world_out.as_ref().map(|w| w.surprise).unwrap_or(0.0),
                ),
                uncertainty_q: ucf_types::UQ0_16::from_f32_clamped(
                    lfm_out.as_ref().map(|l| l.uncertainty).unwrap_or(1.0),
                ),
                coherence_q: None,
                nsr_risk_q: None,
            },
            candidates: candidates
                .iter()
                .map(candidate_feature_from_decision)
                .collect(),
        };
        counters.ebm_candidates_scored_total = counters
            .ebm_candidates_scored_total
            .saturating_add(ebm_input.candidates.len() as u64);
        let mut ebm_budget = ucf_compute::work_meter::WorkMeter::new(64);
        let _ = ebm.score_candidates(ebm_input, &mut ebm_budget);
        stage_stats
            .entry("ebm".to_string())
            .or_default()
            .push(t_ebm.elapsed().as_secs_f64() * 1000.0);

        let t6 = Instant::now();
        let _ = pack.llm().infer(
            &LlmRequest {
                schema_version: 1,
                t: tick,
                decision_id: tick,
                candidate_id: 0,
                output_class: LlmOutputClass::SafeText,
                prompt: format!("bench:{}:{}", fixture.intent_summary, sig),
                context_digest: input.context_digest,
                evidence_chain_digest: [0; 32],
                lfm_readout_digest: lfm_out.as_ref().map(|l| l.liquid_readout_digest),
                lfm_uncertainty: lfm_out.as_ref().map(|l| l.uncertainty),
                lfm_stability: lfm_out.as_ref().map(|l| l.stability),
                coherence: None,
                instability: None,
                risk: Some(risk),
                confidence: Some(confidence),
                seed: budget.seed,
                max_tokens: 64,
                temperature: 0.0,
                top_p: 1.0,
                sampling_enabled: false,
            }
            .bounded(),
            budget,
        );
        stage_stats
            .get_mut("llm")
            .expect("stage exists")
            .push(t6.elapsed().as_secs_f64() * 1000.0);

        if world_out
            .as_ref()
            .is_some_and(|world| world.quality == StageQuality::DegradedFallback)
        {
            *counters
                .degraded_stages
                .entry("world_model/step".to_string())
                .or_insert(0) += 1;
        }
        if sae_out
            .as_ref()
            .is_some_and(|sae| sae.quality == StageQuality::DegradedFallback)
        {
            *counters
                .degraded_stages
                .entry("sae/extract".to_string())
                .or_insert(0) += 1;
        }
        if ssm_out
            .as_ref()
            .is_some_and(|ssm| ssm.quality == StageQuality::DegradedFallback)
        {
            *counters
                .degraded_stages
                .entry("ssm/step".to_string())
                .or_insert(0) += 1;
        }
        if lfm_out
            .as_ref()
            .is_some_and(|lfm| lfm.quality == StageQuality::DegradedFallback)
        {
            *counters
                .degraded_stages
                .entry("lfm/step".to_string())
                .or_insert(0) += 1;
        }

        tick_stats.push(tick_start.elapsed().as_secs_f64() * 1000.0);
        if tick % args.rss_sample_every.max(1) == 0 {
            if let Some(rss_mb) = read_rss_mb() {
                rss_samples.push(rss_mb);
            }
        }
    }

    counters.budget_exceeded_events = counters.degraded_stages.values().copied().sum::<u64>();
    counters.ebm_mean_k_q = if tick_cap > 0 {
        (((counters.ebm_candidates_scored_total as u128) << 16) / (tick_cap as u128))
            .min(u16::MAX as u128) as u16
    } else {
        0
    };

    let elapsed_s = run_start.elapsed().as_secs_f64();
    let throughput = if elapsed_s > 0.0 {
        tick_cap as f64 / elapsed_s
    } else {
        0.0
    };

    counters.tool_issuance_attempts = 0;
    counters.tool_issuance_denies = 0;

    let memory = if rss_samples.is_empty() {
        MemoryStats {
            mode: "unsupported".to_string(),
            samples: 0,
            min_rss_mb: None,
            mean_rss_mb: None,
            max_rss_mb: None,
            cap_mb: args.rss_cap_mb,
            cap_exceeded: false,
        }
    } else {
        let min = rss_samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = rss_samples
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = rss_samples.iter().sum::<f64>() / rss_samples.len() as f64;
        let cap_exceeded = args.rss_cap_mb.is_some_and(|cap| max > cap as f64);
        MemoryStats {
            mode: "linux_proc_status".to_string(),
            samples: rss_samples.len(),
            min_rss_mb: Some(min),
            mean_rss_mb: Some(mean),
            max_rss_mb: Some(max),
            cap_mb: args.rss_cap_mb,
            cap_exceeded,
        }
    };

    let mut hints = Vec::new();
    if let Some(lfm) = stage_stats.get("lfm") {
        let s = lfm.stats();
        if s.p95_ms > (budget.max_micros as f64 / 1000.0) {
            hints.push(
                "lfm p95 exceeds compute budget: raise lfm stage budget or switch kernel profile"
                    .to_string(),
            );
        }
    }
    if let Some(llm) = stage_stats.get("llm") {
        let s = llm.stats();
        if s.p95_ms > 2.0 {
            hints.push(
                "llm dominates tail latency: reduce max_tokens_eff or tighten uncertainty scaling"
                    .to_string(),
            );
        }
    }
    if let Some(ebm_stats) = stage_stats.get("ebm") {
        let s = ebm_stats.stats();
        if s.p95_ms > 1.0 {
            hints.push(
                "ebm p95 exceeds 1ms: reduce candidate K or simplify feature extraction"
                    .to_string(),
            );
        }
    }
    if counters.backpressure_ticks * 3 > tick_cap {
        hints.push(
            "backpressure active on >33% ticks: tune governor tier thresholds or reduce workload"
                .to_string(),
        );
    }

    let mut latency = BTreeMap::new();
    for (k, v) in stage_stats {
        latency.insert(k, v.stats());
    }
    let pack_digest = hex::encode(pack.meta().digest);
    let report = BenchReport {
        schema_version: BENCH_SCHEMA_VERSION,
        run_id,
        scenario_id: fixture.scenario,
        scenario_digest,
        backend_pack_digest: pack_digest,
        ticks: tick_cap,
        throughput_ticks_per_sec: throughput,
        tick_time_ms: tick_stats.stats(),
        stage_latency_ms: latency,
        counters,
        memory,
        hints,
    };

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            scenario: PathBuf::from("fixtures/e2e_scenario_a.json"),
            ticks: 256,
            out: PathBuf::from("./out/bench_report.json"),
            rss_sample_every: DEFAULT_RSS_SAMPLE_EVERY,
            rss_cap_mb: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QuantileEstimator;

    #[test]
    fn quantile_estimator_orders_values() {
        let mut q = QuantileEstimator::default();
        for value in [10.0, 1.0, 5.0, 2.0, 8.0] {
            q.push(value);
        }
        let s = q.stats();
        assert_eq!(s.samples, 5);
        assert!(s.p95_ms >= s.p50_ms);
        assert!(s.p99_ms >= s.p95_ms);
        assert!(s.max_ms >= s.p99_ms);
    }
}
