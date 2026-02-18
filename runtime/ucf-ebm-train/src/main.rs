#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use safetensors::{serialize, tensor::TensorView, Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
enum TrainError {
    #[error("usage: ucf-ebm-train --enable-training-runner --dataset <jsonl> --initial-weights <safetensors> [--steps <n>] [--lr <f32>] [--batch <n>] [--seed <u64>] [--out-dir <path>]")]
    Usage,
    #[error("training runner is disabled unless --enable-training-runner is provided")]
    Disabled,
    #[error("invalid argument: {0}")]
    Invalid(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
}

#[derive(Debug, Clone, Deserialize)]
struct Sample {
    context_digest: String,
    signals_q: SignalsQ,
    candidates: Vec<Candidate>,
    label: Label,
}

#[derive(Debug, Clone, Deserialize)]
struct SignalsQ {
    risk_q: Option<u16>,
    pressure_q: Option<u16>,
    surprise_q: Option<u16>,
    uncertainty_q: Option<u16>,
}

#[derive(Debug, Clone, Deserialize)]
struct Candidate {
    candidate_id: u16,
    intent_kind: u8,
    output_class: u8,
    tool_intent_count: u8,
    allowed: bool,
    policy_hint: u8,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Label {
    ChosenCandidate {
        chosen_candidate_id: u16,
    },
    PairwisePreference {
        better_candidate_id: u16,
        worse_candidate_id: u16,
    },
}

#[derive(Debug, Clone, Serialize)]
struct TrainingConfig {
    steps: u32,
    lr: f32,
    batch: u32,
    seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TrainingReport {
    schema_version: u16,
    dataset_path: String,
    dataset_digest: String,
    dataset_samples: usize,
    initial_weight_hash: String,
    output_weight_hash: String,
    config_digest: String,
    loss_summary: LossSummary,
    nondet_risk: bool,
}

#[derive(Debug, Clone, Serialize)]
struct LossSummary {
    first_loss: f32,
    last_loss: f32,
    min_loss: f32,
    max_loss: f32,
    points: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Weights {
    d: usize,
    h: usize,
    w1: Vec<f32>, // d x h
    b1: Vec<f32>, // h
    w2: Vec<f32>, // h
    b2: f32,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), TrainError> {
    let args: Vec<String> = std::env::args().collect();
    if !has_flag(&args, "--enable-training-runner") {
        return Err(TrainError::Disabled);
    }
    let dataset = arg_value(&args, "--dataset").ok_or(TrainError::Usage)?;
    let initial_weights = arg_value(&args, "--initial-weights").ok_or(TrainError::Usage)?;
    let cfg = TrainingConfig {
        steps: parse_u32(&args, "--steps", 10).min(1000),
        lr: parse_f32(&args, "--lr", 0.0005),
        batch: parse_u32(&args, "--batch", 16).clamp(1, 64),
        seed: parse_u64(&args, "--seed", 7),
    };
    let out_dir = arg_value(&args, "--out-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./out"));
    if !is_allowed_output_root(&out_dir) {
        return Err(TrainError::Invalid(format!(
            "out-dir {} not allowed; use ./out or ./models/staging",
            out_dir.display()
        )));
    }

    let mut samples = load_dataset(Path::new(&dataset))?;
    samples.sort_by(|a, b| match a.context_digest.cmp(&b.context_digest) {
        Ordering::Equal => a.candidates.len().cmp(&b.candidates.len()),
        o => o,
    });
    if samples.is_empty() {
        return Err(TrainError::Invalid("dataset has no samples".to_string()));
    }

    let initial_bytes = fs::read(&initial_weights)?;
    let initial_hash = hex::encode(Sha256::digest(&initial_bytes));
    let mut weights = load_weights(&initial_bytes)?;
    let losses = train(&samples, &mut weights, &cfg);

    fs::create_dir_all(&out_dir)?;
    fs::create_dir_all("./models/staging")?;
    let staged_tmp = PathBuf::from("./models/staging/ebm_tmp.safetensors");
    let out_bytes = encode_weights(&weights)?;
    fs::write(&staged_tmp, &out_bytes)?;
    let output_hash = hex::encode(Sha256::digest(&out_bytes));
    let staged_final = PathBuf::from(format!("./models/staging/ebm_{output_hash}.safetensors"));
    fs::rename(&staged_tmp, &staged_final)?;

    let report = TrainingReport {
        schema_version: 1,
        dataset_path: dataset.clone(),
        dataset_digest: digest_file(Path::new(&dataset))?,
        dataset_samples: samples.len(),
        initial_weight_hash: initial_hash,
        output_weight_hash: output_hash.clone(),
        config_digest: digest_json(&cfg)?,
        loss_summary: summarize_losses(&losses),
        nondet_risk: false,
    };
    let report_path = out_dir.join("ebm_training_report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;

    println!("weights={}", staged_final.display());
    println!("report={}", report_path.display());
    Ok(())
}

fn load_dataset(path: &Path) -> Result<Vec<Sample>, TrainError> {
    let body = fs::read_to_string(path)?;
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(TrainError::from)
}

fn train(samples: &[Sample], weights: &mut Weights, cfg: &TrainingConfig) -> Vec<f32> {
    let mut rng = cfg.seed;
    let mut losses = Vec::new();
    for _ in 0..cfg.steps {
        let mut step_loss = 0.0f32;
        for _ in 0..cfg.batch {
            let idx = (lcg_next(&mut rng) as usize) % samples.len();
            step_loss += train_one(&samples[idx], weights, cfg.lr);
        }
        losses.push(step_loss / (cfg.batch as f32));
    }
    losses
}

fn train_one(sample: &Sample, w: &mut Weights, lr: f32) -> f32 {
    let (better, worse) = pair_from_label(sample);
    let x_good = feature_vector(sample, better, w.d);
    let x_bad = feature_vector(sample, worse, w.d);
    let (s_good, z_good, a_good) = forward(w, &x_good);
    let (s_bad, z_bad, a_bad) = forward(w, &x_bad);
    let margin = s_good - s_bad;
    let sig = 1.0 / (1.0 + (-margin).exp());

    let dsg = sig;
    let dsb = -sig;

    for j in 0..w.h {
        w.w2[j] -= lr * (dsg * a_good[j] + dsb * a_bad[j]);
    }
    w.b2 -= lr * (dsg + dsb);

    let mut da_good = vec![0.0f32; w.h];
    let mut da_bad = vec![0.0f32; w.h];
    for j in 0..w.h {
        da_good[j] = dsg * w.w2[j] * if z_good[j] > 0.0 { 1.0 } else { 0.0 };
        da_bad[j] = dsb * w.w2[j] * if z_bad[j] > 0.0 { 1.0 } else { 0.0 };
    }

    for i in 0..w.d {
        for j in 0..w.h {
            let grad = da_good[j] * x_good[i] + da_bad[j] * x_bad[i];
            w.w1[i * w.h + j] -= lr * grad;
        }
    }
    for j in 0..w.h {
        w.b1[j] -= lr * (da_good[j] + da_bad[j]);
    }

    (1.0 + margin.exp()).ln()
}

fn pair_from_label(sample: &Sample) -> (u16, u16) {
    match sample.label {
        Label::PairwisePreference {
            better_candidate_id,
            worse_candidate_id,
        } => (better_candidate_id, worse_candidate_id),
        Label::ChosenCandidate {
            chosen_candidate_id,
        } => {
            let alt = sample
                .candidates
                .iter()
                .find(|c| c.candidate_id != chosen_candidate_id)
                .map(|c| c.candidate_id)
                .unwrap_or(chosen_candidate_id);
            (chosen_candidate_id, alt)
        }
    }
}

fn feature_vector(sample: &Sample, cid: u16, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    let candidate = sample
        .candidates
        .iter()
        .find(|c| c.candidate_id == cid)
        .or_else(|| sample.candidates.first());
    let (allowed, tool_count, policy_hint, intent_kind, output_class) = if let Some(c) = candidate {
        (
            if c.allowed { 1.0 } else { 0.0 },
            (c.tool_intent_count as f32) / 8.0,
            (c.policy_hint as f32) / 255.0,
            (c.intent_kind as f32) / 255.0,
            (c.output_class as f32) / 255.0,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };

    let signal = |v: Option<u16>| v.map(|q| (q as f32) / 10_000.0).unwrap_or(0.0);
    let basis = [
        allowed,
        tool_count,
        policy_hint,
        intent_kind,
        output_class,
        signal(sample.signals_q.risk_q),
        signal(sample.signals_q.pressure_q),
        signal(sample.signals_q.surprise_q),
        signal(sample.signals_q.uncertainty_q),
    ];
    while out.len() < dim {
        out.push(basis[out.len() % basis.len()]);
    }
    out
}

fn forward(w: &Weights, x: &[f32]) -> (f32, Vec<f32>, Vec<f32>) {
    let mut z = vec![0.0f32; w.h];
    let mut a = vec![0.0f32; w.h];
    for j in 0..w.h {
        let mut v = w.b1[j];
        for (i, xv) in x.iter().take(w.d).enumerate() {
            v += *xv * w.w1[i * w.h + j];
        }
        z[j] = v;
        a[j] = v.max(0.0);
    }
    let mut s = w.b2;
    for (j, aj) in a.iter().enumerate() {
        s += aj * w.w2[j];
    }
    (s, z, a)
}

fn load_weights(bytes: &[u8]) -> Result<Weights, TrainError> {
    let safetensors = SafeTensors::deserialize(bytes)?;
    let w1 = safetensors.tensor("ebm.w1")?;
    let b1 = safetensors.tensor("ebm.b1")?;
    let w2 = safetensors.tensor("ebm.w2")?;
    let b2 = safetensors.tensor("ebm.b2")?;

    let shape = w1.shape();
    if shape.len() != 2 {
        return Err(TrainError::Invalid("ebm.w1 must be rank-2".to_string()));
    }
    let d = shape[0];
    let h = shape[1];
    if b1.shape() != [h] || w2.shape() != [h, 1] || b2.shape() != [1] {
        return Err(TrainError::Invalid("invalid ebm tensor shapes".to_string()));
    }

    let b2_values = bytes_to_f32(b2.data())?;

    Ok(Weights {
        d,
        h,
        w1: bytes_to_f32(w1.data())?,
        b1: bytes_to_f32(b1.data())?,
        w2: bytes_to_f32(w2.data())?,
        b2: *b2_values
            .first()
            .ok_or_else(|| TrainError::Invalid("ebm.b2 missing value".to_string()))?,
    })
}

fn encode_weights(weights: &Weights) -> Result<Vec<u8>, TrainError> {
    let w1_bytes = f32_to_bytes(&weights.w1);
    let b1_bytes = f32_to_bytes(&weights.b1);
    let w2_bytes = f32_to_bytes(&weights.w2);
    let b2_bytes = f32_to_bytes(&[weights.b2]);

    let mut tensors: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
    tensors.insert(
        "ebm.w1".to_string(),
        TensorView::new(Dtype::F32, vec![weights.d, weights.h], &w1_bytes)?,
    );
    tensors.insert(
        "ebm.b1".to_string(),
        TensorView::new(Dtype::F32, vec![weights.h], &b1_bytes)?,
    );
    tensors.insert(
        "ebm.w2".to_string(),
        TensorView::new(Dtype::F32, vec![weights.h, 1], &w2_bytes)?,
    );
    tensors.insert(
        "ebm.b2".to_string(),
        TensorView::new(Dtype::F32, vec![1], &b2_bytes)?,
    );
    serialize(tensors, &None).map_err(TrainError::from)
}

fn summarize_losses(losses: &[f32]) -> LossSummary {
    let first = losses.first().copied().unwrap_or(0.0);
    let last = losses.last().copied().unwrap_or(0.0);
    let min = losses
        .iter()
        .copied()
        .fold(f32::INFINITY, |a, b| if b < a { b } else { a });
    let max = losses
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    LossSummary {
        first_loss: first,
        last_loss: last,
        min_loss: if min.is_finite() { min } else { 0.0 },
        max_loss: if max.is_finite() { max } else { 0.0 },
        points: losses.iter().copied().take(64).collect(),
    }
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == key {
            return args.get(i + 1).cloned();
        }
        i += 1;
    }
    None
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn parse_u32(args: &[String], key: &str, default: u32) -> u32 {
    arg_value(args, key)
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

fn parse_u64(args: &[String], key: &str, default: u64) -> u64 {
    arg_value(args, key)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_f32(args: &[String], key: &str, default: f32) -> f32 {
    arg_value(args, key)
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, TrainError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(TrainError::Invalid(
            "f32 tensor bytes not divisible by 4".to_string(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>()
}

fn digest_file(path: &Path) -> Result<String, TrainError> {
    let bytes = fs::read(path)?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn digest_json<T: Serialize>(value: &T) -> Result<String, TrainError> {
    let bytes = serde_json::to_vec(value)?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn lcg_next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn is_allowed_output_root(path: &Path) -> bool {
    let normalized = normalize_rel(path);
    normalized.starts_with(Path::new("out")) || normalized.starts_with(Path::new("models/staging"))
}

fn normalize_rel(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::Normal(part) => out.push(part),
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir | Component::Prefix(_) => {}
        }
    }
    out
}
