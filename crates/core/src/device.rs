use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    DiscreteGpu,
    IntegratedGpu,
    CpuCompute,
    VirtualDevice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuArchitecture {
    Rdna1,
    Rdna2,
    Rdna3,
    VegaGcn5,
    IntelXe,
    NvidiaAmpere,
    NvidiaAda,
    GenericVulkan,
    HostCpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub api_version_major: u32,
    pub api_version_minor: u32,
    pub api_version_patch: u32,
    pub driver_version: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub pci_bus_id: Option<String>,
    pub supports_timeline_semaphores: bool,
    pub supports_external_memory_fd: bool,
    pub supports_buffer_device_address: bool,
    pub supports_memory_budget: bool,
    pub async_compute_queues: u32,
    pub transfer_queues: u32,
    pub native_wavefront_size: u32,
    pub estimated_fp32_tflops: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub name: String,
    pub kind: DeviceKind,
    pub architecture: GpuArchitecture,
    pub capabilities: DeviceCapabilities,
}
