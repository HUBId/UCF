use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackendKind {
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuCapability {
    pub available: bool,
    pub backend: Option<GpuBackendKind>,
}

pub fn detect_gpu_capability() -> GpuCapability {
    if let Ok(v) = std::env::var("UCF_GPU_AVAILABLE") {
        let available = matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes");
        return GpuCapability {
            available,
            backend: if available { default_backend() } else { None },
        };
    }
    GpuCapability {
        available: false,
        backend: None,
    }
}

fn default_backend() -> Option<GpuBackendKind> {
    #[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
    {
        Some(GpuBackendKind::Cuda)
    }
    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    {
        Some(GpuBackendKind::Metal)
    }
    #[cfg(not(any(
        all(feature = "gpu-cuda", target_os = "linux"),
        all(feature = "gpu-metal", target_os = "macos")
    )))]
    {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuResourceCaps {
    pub max_vram_bytes: u64,
    pub max_kernel_micros: u64,
}

impl GpuResourceCaps {
    pub fn from_env() -> Self {
        Self {
            max_vram_bytes: std::env::var("UCF_GPU_MAX_VRAM_BYTES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512 * 1024 * 1024),
            max_kernel_micros: std::env::var("UCF_GPU_MAX_KERNEL_MICROS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100_000),
        }
    }

    pub fn within_caps(&self, estimated_vram_bytes: u64, estimated_kernel_micros: u64) -> bool {
        estimated_vram_bytes <= self.max_vram_bytes
            && estimated_kernel_micros <= self.max_kernel_micros
    }
}
