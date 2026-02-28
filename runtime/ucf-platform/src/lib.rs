#![forbid(unsafe_code)]

use std::time::Instant;

use serde::{Deserialize, Serialize};

pub trait PlatformProbe {
    fn probe() -> PlatformInfo;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OsKind {
    Windows,
    Linux,
    Macos,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CpuArch {
    X86_64,
    Aarch64,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorKind {
    None,
    Cuda,
    Metal,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlatformInfo {
    pub os: OsKind,
    pub cpu_cores: Option<u16>,
    pub cpu_arch: CpuArch,
    pub mem_total_mb: Option<u64>,
    pub accel: AcceleratorKind,
    pub monotonic_clock_ok: bool,
}

impl PlatformInfo {
    pub fn summary(&self) -> String {
        format!(
            "os={:?} arch={:?} cores={} mem_mb={} accel={:?} monotonic_clock_ok={}",
            self.os,
            self.cpu_arch,
            self.cpu_cores
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.mem_total_mb
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.accel,
            self.monotonic_clock_ok,
        )
    }
}

pub struct LocalPlatformProbe;

impl PlatformProbe for LocalPlatformProbe {
    fn probe() -> PlatformInfo {
        PlatformInfo {
            os: detect_os(),
            cpu_cores: detect_cpu_cores(),
            cpu_arch: detect_cpu_arch(),
            mem_total_mb: detect_mem_total_mb(),
            accel: detect_accel(),
            monotonic_clock_ok: monotonic_clock_ok(),
        }
    }
}

fn detect_os() -> OsKind {
    match std::env::consts::OS {
        "windows" => OsKind::Windows,
        "linux" => OsKind::Linux,
        "macos" => OsKind::Macos,
        _ => OsKind::Other,
    }
}

fn detect_cpu_arch() -> CpuArch {
    match std::env::consts::ARCH {
        "x86_64" => CpuArch::X86_64,
        "aarch64" => CpuArch::Aarch64,
        _ => CpuArch::Other,
    }
}

fn detect_cpu_cores() -> Option<u16> {
    let cores = std::thread::available_parallelism().ok()?.get();
    u16::try_from(cores).ok()
}

fn detect_mem_total_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = raw.lines().find(|line| line.starts_with("MemTotal:"))?;
        let kb = line
            .split_whitespace()
            .nth(1)
            .and_then(|v| v.parse::<u64>().ok())?;
        Some(kb / 1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn detect_accel() -> AcceleratorKind {
    if let Ok(value) = std::env::var("UCF_ACCEL_CAPABILITY") {
        return parse_accel(&value);
    }

    #[cfg(feature = "cuda")]
    {
        return AcceleratorKind::Cuda;
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return AcceleratorKind::Metal;
    }
    AcceleratorKind::None
}

fn parse_accel(value: &str) -> AcceleratorKind {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => AcceleratorKind::None,
        "cuda" => AcceleratorKind::Cuda,
        "metal" => AcceleratorKind::Metal,
        _ => AcceleratorKind::Other,
    }
}

fn monotonic_clock_ok() -> bool {
    let start = Instant::now();
    let delta = start.elapsed();
    delta <= std::time::Duration::from_secs(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_info_json_is_stable_shape() {
        let info = PlatformInfo {
            os: OsKind::Linux,
            cpu_cores: Some(8),
            cpu_arch: CpuArch::X86_64,
            mem_total_mb: Some(1024),
            accel: AcceleratorKind::None,
            monotonic_clock_ok: true,
        };
        let json = serde_json::to_string(&info).expect("serialize");
        assert_eq!(json, "{\"os\":\"linux\",\"cpu_cores\":8,\"cpu_arch\":\"x86_64\",\"mem_total_mb\":1024,\"accel\":\"none\",\"monotonic_clock_ok\":true}");
    }

    #[test]
    fn probe_is_best_effort() {
        let info = LocalPlatformProbe::probe();
        assert!(matches!(
            info.os,
            OsKind::Windows | OsKind::Linux | OsKind::Macos | OsKind::Other
        ));
    }
}
