use std::collections::BTreeMap;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::backend_pack::{
    BackendComponentId, BackendPack, BackendPackId, BackendPackMeta, FixtureManager,
    ModelSlotProvenance,
};
use crate::capabilities::{
    LlmInference, LlmRequest, LlmResponse, LlmStubBackend, SaeExtractor, WorldModelPredictor,
};
use crate::feature_extractor::{SaeInput, SaeOutput, ToySaeExtractor};
use crate::ipc::{
    read_frame, write_frame, ComputeRequest, ComputeResponse, ComputeStatus, InitRequest,
    StageInput, StageOutput, WorkerRequest, WorkerResponse, WorkerStage, IPC_SCHEMA_VERSION,
};
use crate::lfm::{LfmInput, LfmKernel, LfmOutput, ToyLfmKernel};
use crate::ssm::{SsmInput, SsmKernel, SsmOutput, ToySsmKernel};
use crate::world_model::{MockJepaPredictor, StageQuality, WorldModelInput, WorldModelOutput};
use crate::{CodeVersionTag, ComputeBudget, ComputeError, ModelStore};

#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkerSpawnRecord {
    pub stage: WorkerStage,
    pub pid: u32,
    pub version: u16,
    pub model_hashes_digest: [u8; 32],
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkerKillRecord {
    pub stage: WorkerStage,
    pub reason: String,
    pub t: u64,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkerRestartRecord {
    pub stage: WorkerStage,
    pub count: u32,
    pub since: u64,
    pub reason: String,
}
#[derive(Debug, Default)]
pub struct WorkerAuditLog {
    pub spawns: Vec<WorkerSpawnRecord>,
    pub kills: Vec<WorkerKillRecord>,
    pub restarts: Vec<WorkerRestartRecord>,
}

struct WorkerHandle {
    child: Child,
    stdin: ChildStdin,
    response_rx: mpsc::Receiver<WorkerResponse>,
    restart_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerFailureClass {
    DispatchFailedBeforeExecution,
    TransportFailure,
    WorkerUnavailableOrStale,
    WorkerExecutionCrashed,
    StructuredExecutionFailure,
}

impl WorkerFailureClass {
    fn retryable(self) -> bool {
        matches!(
            self,
            Self::DispatchFailedBeforeExecution
                | Self::TransportFailure
                | Self::WorkerUnavailableOrStale
                | Self::WorkerExecutionCrashed
        )
    }

    fn as_code(self) -> &'static str {
        match self {
            Self::DispatchFailedBeforeExecution => "worker_dispatch_failed_before_execution",
            Self::TransportFailure => "worker_transport_failure",
            Self::WorkerUnavailableOrStale => "worker_unavailable_or_stale",
            Self::WorkerExecutionCrashed => "worker_execution_crashed",
            Self::StructuredExecutionFailure => "worker_structured_execution_failure",
        }
    }
}

pub struct WorkerManager {
    workers: Mutex<BTreeMap<WorkerStage, WorkerHandle>>,
    audit: Mutex<WorkerAuditLog>,
    seed: u64,
    request_id: AtomicU64,
    model_hashes_digest: [u8; 32],
}

impl WorkerManager {
    const MAX_TRANSIENT_RETRIES: u8 = 1;

    pub fn new(seed: u64, model_hashes_digest: [u8; 32]) -> Self {
        Self {
            workers: Mutex::new(BTreeMap::new()),
            audit: Mutex::new(WorkerAuditLog::default()),
            seed,
            request_id: AtomicU64::new(1),
            model_hashes_digest,
        }
    }

    fn ensure_worker(&self, stage: WorkerStage) -> Result<(), ComputeError> {
        let mut workers = self.workers.lock().map_err(|_| ComputeError::Internal {
            reason: "worker mutex poisoned".to_string(),
        })?;
        if workers.contains_key(&stage) {
            return Ok(());
        }
        let bin = std::env::var("UCF_WORKER_BIN").unwrap_or_else(|_| "ucf-worker".to_string());
        let mut child = Command::new(bin)
            .env(
                "UCF_WORKER_STAGE",
                format!("{:?}", stage).to_ascii_lowercase(),
            )
            .env("UCF_WORKER_NETWORK", "disabled")
            .env(
                "UCF_WORKER_MEMORY_LIMIT_MB",
                std::env::var("UCF_WORKER_MEMORY_LIMIT_MB").unwrap_or_else(|_| "512".to_string()),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| ComputeError::Internal {
                reason: format!("spawn worker failed: {e}"),
            })?;
        let pid = child.id();
        let mut stdin = child.stdin.take().ok_or_else(|| ComputeError::Internal {
            reason: "worker stdin missing".to_string(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| ComputeError::Internal {
            reason: "worker stdout missing".to_string(),
        })?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stdout);
            while let Ok(msg) = read_frame::<_, WorkerResponse>(&mut reader) {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });

        write_frame(
            &mut stdin,
            &WorkerRequest::Init(InitRequest {
                schema_version: IPC_SCHEMA_VERSION,
                stage,
                model_hashes_digest: self.model_hashes_digest,
            }),
        )
        .map_err(|e| ComputeError::Internal {
            reason: format!("init write: {e}"),
        })?;

        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(WorkerResponse::InitAck { schema_version })
                if schema_version == IPC_SCHEMA_VERSION => {}
            Ok(other) => {
                return Err(ComputeError::Internal {
                    reason: format!("unexpected init response: {other:?}"),
                })
            }
            Err(e) => {
                return Err(ComputeError::Internal {
                    reason: format!("worker init timeout: {e}"),
                })
            }
        }

        workers.insert(
            stage,
            WorkerHandle {
                child,
                stdin,
                response_rx: rx,
                restart_count: 0,
            },
        );
        if let Ok(mut audit) = self.audit.lock() {
            audit.spawns.push(WorkerSpawnRecord {
                stage,
                pid,
                version: IPC_SCHEMA_VERSION,
                model_hashes_digest: self.model_hashes_digest,
            });
        }
        Ok(())
    }

    fn restart_worker(&self, stage: WorkerStage, reason: &str, t: u64) {
        if let Ok(mut workers) = self.workers.lock() {
            if let Some(mut h) = workers.remove(&stage) {
                let _ = h.child.kill();
                h.restart_count = h.restart_count.saturating_add(1);
                if let Ok(mut audit) = self.audit.lock() {
                    audit.kills.push(WorkerKillRecord {
                        stage,
                        reason: reason.to_string(),
                        t,
                    });
                    audit.restarts.push(WorkerRestartRecord {
                        stage,
                        count: h.restart_count,
                        since: t,
                        reason: reason.to_string(),
                    });
                }
            }
        }
        let _ = self.ensure_worker(stage);
    }

    pub fn compute(
        &self,
        stage: WorkerStage,
        t: u64,
        timeout_ms: u32,
        input: StageInput,
    ) -> Result<StageOutput, ComputeError> {
        let mut attempts = 0_u8;
        loop {
            attempts = attempts.saturating_add(1);
            match self.compute_once(stage, t, timeout_ms, input.clone()) {
                Ok(output) => return Ok(output),
                Err((failure_class, err)) => {
                    if !failure_class.retryable() || attempts > Self::MAX_TRANSIENT_RETRIES + 1 {
                        return Err(err);
                    }
                    self.restart_worker(stage, failure_class.as_code(), t);
                }
            }
        }
    }

    fn compute_once(
        &self,
        stage: WorkerStage,
        t: u64,
        timeout_ms: u32,
        input: StageInput,
    ) -> Result<StageOutput, (WorkerFailureClass, ComputeError)> {
        self.ensure_worker(stage).map_err(|err| {
            (
                WorkerFailureClass::WorkerUnavailableOrStale,
                prefix_worker_error(
                    WorkerFailureClass::WorkerUnavailableOrStale,
                    err.to_string(),
                ),
            )
        })?;
        let req_id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let req = WorkerRequest::Compute(Box::new(ComputeRequest {
            schema_version: IPC_SCHEMA_VERSION,
            request_id: req_id,
            t,
            stage,
            seed: self.seed ^ ((stage as u8 as u64) * 0x9E37),
            timeout_ms,
            input,
        }));
        let mut workers = self.workers.lock().map_err(|_| {
            (
                WorkerFailureClass::DispatchFailedBeforeExecution,
                prefix_worker_error(
                    WorkerFailureClass::DispatchFailedBeforeExecution,
                    "worker mutex poisoned".to_string(),
                ),
            )
        })?;
        let handle = workers.get_mut(&stage).ok_or_else(|| {
            (
                WorkerFailureClass::DispatchFailedBeforeExecution,
                prefix_worker_error(
                    WorkerFailureClass::DispatchFailedBeforeExecution,
                    "worker missing".to_string(),
                ),
            )
        })?;
        write_frame(&mut handle.stdin, &req).map_err(|e| {
            (
                WorkerFailureClass::DispatchFailedBeforeExecution,
                prefix_worker_error(
                    WorkerFailureClass::DispatchFailedBeforeExecution,
                    format!("request write: {e}"),
                ),
            )
        })?;
        match handle
            .response_rx
            .recv_timeout(Duration::from_millis(u64::from(timeout_ms)))
        {
            Ok(WorkerResponse::Compute(resp)) => {
                let resp = *resp;
                if resp.schema_version != IPC_SCHEMA_VERSION || resp.request_id != req_id {
                    return Err((
                        WorkerFailureClass::TransportFailure,
                        prefix_worker_error(
                            WorkerFailureClass::TransportFailure,
                            "schema/request mismatch".to_string(),
                        ),
                    ));
                }
                match resp.status {
                    ComputeStatus::Ok => resp.output.ok_or_else(|| {
                        (
                            WorkerFailureClass::TransportFailure,
                            prefix_worker_error(
                                WorkerFailureClass::TransportFailure,
                                "missing output".to_string(),
                            ),
                        )
                    }),
                    ComputeStatus::Timeout => {
                        drop(workers);
                        self.restart_worker(stage, "timeout", t);
                        Err((
                            WorkerFailureClass::WorkerExecutionCrashed,
                            ComputeError::BudgetExceeded {
                                stage: "worker/timeout",
                                elapsed_micros: u64::from(timeout_ms) * 1000,
                                limit_micros: u64::from(timeout_ms) * 1000,
                            },
                        ))
                    }
                    ComputeStatus::Error => Err((
                        WorkerFailureClass::StructuredExecutionFailure,
                        prefix_worker_error(
                            WorkerFailureClass::StructuredExecutionFailure,
                            resp.error_code
                                .unwrap_or_else(|| "worker_error".to_string()),
                        ),
                    )),
                }
            }
            Ok(other) => Err((
                WorkerFailureClass::TransportFailure,
                prefix_worker_error(
                    WorkerFailureClass::TransportFailure,
                    format!("unexpected response: {other:?}"),
                ),
            )),
            Err(_) => {
                drop(workers);
                self.restart_worker(stage, "timeout", t);
                Err((
                    WorkerFailureClass::WorkerExecutionCrashed,
                    ComputeError::BudgetExceeded {
                        stage: "worker/timeout",
                        elapsed_micros: u64::from(timeout_ms) * 1000,
                        limit_micros: u64::from(timeout_ms) * 1000,
                    },
                ))
            }
        }
    }
}

fn prefix_worker_error(class: WorkerFailureClass, detail: String) -> ComputeError {
    ComputeError::Internal {
        reason: format!("{}:{detail}", class.as_code()),
    }
}

struct WorkerWorld {
    manager: Arc<WorkerManager>,
}
impl WorldModelPredictor for WorkerWorld {
    fn name(&self) -> &'static str {
        "worker_world_v1"
    }
    fn step(
        &mut self,
        input: &WorldModelInput,
        budget: ComputeBudget,
    ) -> Result<WorldModelOutput, ComputeError> {
        match self.manager.compute(
            WorkerStage::World,
            input.t,
            (budget.max_micros / 1000) as u32 + 1,
            StageInput::World(input.clone()),
        )? {
            StageOutput::World(v) => Ok(v),
            _ => Err(ComputeError::Internal {
                reason: "stage output mismatch".to_string(),
            }),
        }
    }
}

struct WorkerSae {
    manager: Arc<WorkerManager>,
}
impl SaeExtractor for WorkerSae {
    fn name(&self) -> &'static str {
        "worker_sae_v1"
    }
    fn extract(&self, input: &SaeInput, budget: ComputeBudget) -> Result<SaeOutput, ComputeError> {
        match self.manager.compute(
            WorkerStage::Sae,
            input.t,
            (budget.max_micros / 1000) as u32 + 1,
            StageInput::Sae(input.clone()),
        )? {
            StageOutput::Sae(v) => Ok(v),
            _ => Err(ComputeError::Internal {
                reason: "stage output mismatch".to_string(),
            }),
        }
    }
}

struct WorkerSsm {
    manager: Arc<WorkerManager>,
}
impl SsmKernel for WorkerSsm {
    fn name(&self) -> &'static str {
        "worker_ssm_v1"
    }
    fn step(&mut self, input: &SsmInput, budget: ComputeBudget) -> Result<SsmOutput, ComputeError> {
        match self.manager.compute(
            WorkerStage::Ssm,
            input.t,
            (budget.max_micros / 1000) as u32 + 1,
            StageInput::Ssm(*input),
        )? {
            StageOutput::Ssm(v) => Ok(v),
            _ => Err(ComputeError::Internal {
                reason: "stage output mismatch".to_string(),
            }),
        }
    }
}

struct WorkerLfm {
    manager: Arc<WorkerManager>,
}
impl LfmKernel for WorkerLfm {
    fn name(&self) -> &'static str {
        "worker_lfm_v1"
    }
    fn reset_session(&mut self, _seed: u64) {}
    fn step(&mut self, input: &LfmInput, budget: ComputeBudget) -> Result<LfmOutput, ComputeError> {
        match self.manager.compute(
            WorkerStage::Lfm,
            input.t,
            (budget.max_micros / 1000) as u32 + 1,
            StageInput::Lfm(*input),
        )? {
            StageOutput::Lfm(v) => Ok(*v),
            _ => Err(ComputeError::Internal {
                reason: "stage output mismatch".to_string(),
            }),
        }
    }
}

struct WorkerLlm {
    manager: Arc<WorkerManager>,
}
impl LlmInference for WorkerLlm {
    fn name(&self) -> &'static str {
        "worker_llm_v1"
    }
    fn infer(&self, req: &LlmRequest, budget: ComputeBudget) -> Result<LlmResponse, ComputeError> {
        match self.manager.compute(
            WorkerStage::Llm,
            req.t,
            (budget.max_micros / 1000) as u32 + 1,
            StageInput::Llm(req.clone().into()),
        )? {
            StageOutput::Llm(v) => Ok(v.into()),
            _ => Err(ComputeError::Internal {
                reason: "stage output mismatch".to_string(),
            }),
        }
    }
}

pub struct WorkerBackendPack {
    meta: BackendPackMeta,
    slot_provenance: Vec<ModelSlotProvenance>,
    llm: Arc<dyn LlmInference + Send + Sync>,
    world: Mutex<Box<dyn WorldModelPredictor + Send + Sync>>,
    sae: Arc<dyn SaeExtractor + Send + Sync>,
    ssm: Mutex<Box<dyn SsmKernel + Send + Sync>>,
    lfm: Mutex<Box<dyn LfmKernel + Send + Sync>>,
}

impl WorkerBackendPack {
    pub fn build(seed: u64) -> Result<Arc<dyn BackendPack>, ComputeError> {
        let fixtures_digest = FixtureManager.overall_digest()?;
        let model_hashes_digest = ModelStore::from_env_default()
            .map(|s| s.model_hashes_digest())
            .unwrap_or([0; 32]);
        let manager = Arc::new(WorkerManager::new(seed, model_hashes_digest));
        let mut meta = BackendPackMeta {
            schema_version: 1,
            pack_name: "worker_v1",
            pack_id: BackendPackId(6),
            llm_backend: BackendComponentId::ToyV1,
            world_backend: BackendComponentId::ToyV1,
            sae_backend: BackendComponentId::ToyV1,
            ssm_backend: BackendComponentId::ToyV1,
            lfm_backend: BackendComponentId::ToyV1,
            fixtures_digest,
            model_hashes_digest,
            code_version: CodeVersionTag::current(),
            digest: [0; 32],
        };
        meta.digest = meta.canonical_digest();
        Ok(Arc::new(Self {
            meta,
            slot_provenance: Vec::new(),
            llm: Arc::new(WorkerLlm {
                manager: manager.clone(),
            }),
            world: Mutex::new(Box::new(WorkerWorld {
                manager: manager.clone(),
            })),
            sae: Arc::new(WorkerSae {
                manager: manager.clone(),
            }),
            ssm: Mutex::new(Box::new(WorkerSsm {
                manager: manager.clone(),
            })),
            lfm: Mutex::new(Box::new(WorkerLfm { manager })),
        }))
    }
}

impl BackendPack for WorkerBackendPack {
    fn meta(&self) -> &BackendPackMeta {
        &self.meta
    }
    fn model_slot_provenance(&self) -> &[ModelSlotProvenance] {
        &self.slot_provenance
    }
    fn llm(&self) -> &dyn LlmInference {
        self.llm.as_ref()
    }
    fn world(&self) -> &Mutex<Box<dyn WorldModelPredictor + Send + Sync>> {
        &self.world
    }
    fn sae(&self) -> &dyn SaeExtractor {
        self.sae.as_ref()
    }
    fn ssm(&self) -> &Mutex<Box<dyn SsmKernel + Send + Sync>> {
        &self.ssm
    }
    fn lfm(&self) -> &Mutex<Box<dyn LfmKernel + Send + Sync>> {
        &self.lfm
    }
}

pub fn run_worker() -> Result<(), String> {
    let stage = match std::env::var("UCF_WORKER_STAGE")
        .unwrap_or_default()
        .as_str()
    {
        "llm" => WorkerStage::Llm,
        "world" => WorkerStage::World,
        "sae" => WorkerStage::Sae,
        "ssm" => WorkerStage::Ssm,
        "lfm" => WorkerStage::Lfm,
        other => return Err(format!("invalid UCF_WORKER_STAGE={other}")),
    };

    #[cfg(target_os = "linux")]
    {
        if let Ok(mem_mb) = std::env::var("UCF_WORKER_MEMORY_LIMIT_MB") {
            if let Ok(v) = mem_mb.parse::<u64>() {
                let lim = libc::rlimit {
                    rlim_cur: v.saturating_mul(1024 * 1024),
                    rlim_max: v.saturating_mul(1024 * 1024),
                };
                unsafe {
                    let _ = libc::setrlimit(libc::RLIMIT_AS, &lim);
                }
            }
        }
    }

    let llm = LlmStubBackend;
    let mut world = MockJepaPredictor::default();
    let sae = ToySaeExtractor::default();
    let mut ssm = ToySsmKernel::default();
    let mut lfm = ToyLfmKernel::default();

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = std::io::BufReader::new(stdin.lock());
    let mut writer = std::io::BufWriter::new(stdout.lock());

    loop {
        let req = read_frame::<_, WorkerRequest>(&mut reader)?;
        match req {
            WorkerRequest::Init(init) => {
                if init.schema_version != IPC_SCHEMA_VERSION || init.stage != stage {
                    write_frame(
                        &mut writer,
                        &WorkerResponse::Error {
                            schema_version: IPC_SCHEMA_VERSION,
                            error_code: "schema_or_stage_mismatch".to_string(),
                        },
                    )?;
                    continue;
                }
                write_frame(
                    &mut writer,
                    &WorkerResponse::InitAck {
                        schema_version: IPC_SCHEMA_VERSION,
                    },
                )?;
            }
            WorkerRequest::Shutdown => break,
            WorkerRequest::Compute(c) => {
                let c = *c;
                if c.schema_version != IPC_SCHEMA_VERSION || c.stage != stage {
                    write_frame(
                        &mut writer,
                        &WorkerResponse::Compute(Box::new(ComputeResponse {
                            schema_version: IPC_SCHEMA_VERSION,
                            request_id: c.request_id,
                            stage,
                            status: ComputeStatus::Error,
                            elapsed_ms: 0,
                            quality: StageQuality::Unavailable as u8,
                            error_code: Some("schema_or_stage_mismatch".to_string()),
                            output: None,
                        })),
                    )?;
                    continue;
                }
                let start = Instant::now();
                let result = match (stage, c.input) {
                    (WorkerStage::Llm, StageInput::Llm(v)) => llm
                        .infer(&LlmRequest::from(v), ComputeBudget::default())
                        .map(|o| StageOutput::Llm(o.into())),
                    (WorkerStage::World, StageInput::World(v)) => world
                        .step(&v, ComputeBudget::default())
                        .map(StageOutput::World),
                    (WorkerStage::Sae, StageInput::Sae(v)) => sae
                        .extract(&v, ComputeBudget::default())
                        .map(StageOutput::Sae),
                    (WorkerStage::Ssm, StageInput::Ssm(v)) => {
                        ssm.step(&v, ComputeBudget::default()).map(StageOutput::Ssm)
                    }
                    (WorkerStage::Lfm, StageInput::Lfm(v)) => lfm
                        .step(&v, ComputeBudget::default())
                        .map(|o| StageOutput::Lfm(Box::new(o))),
                    _ => Err(ComputeError::InvalidInput {
                        reason: "stage payload mismatch".to_string(),
                    }),
                };
                let elapsed = start.elapsed().as_millis() as u32;
                let resp = match result {
                    Ok(output) => WorkerResponse::Compute(Box::new(ComputeResponse {
                        schema_version: IPC_SCHEMA_VERSION,
                        request_id: c.request_id,
                        stage,
                        status: ComputeStatus::Ok,
                        elapsed_ms: elapsed,
                        quality: StageQuality::Ok as u8,
                        error_code: None,
                        output: Some(output),
                    })),
                    Err(err) => WorkerResponse::Compute(Box::new(ComputeResponse {
                        schema_version: IPC_SCHEMA_VERSION,
                        request_id: c.request_id,
                        stage,
                        status: ComputeStatus::Error,
                        elapsed_ms: elapsed,
                        quality: StageQuality::Unavailable as u8,
                        error_code: Some(err.to_string()),
                        output: None,
                    })),
                };
                write_frame(&mut writer, &resp)?;
            }
        }
    }
    Ok(())
}
