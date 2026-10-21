use super::device::DeviceId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct DomainId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MemoryDomainKind {
    /// Dedicated High-Bandwidth VRAM on discrete GPU (GDDR6)
    DiscreteDedicatedVram,
    /// System-memory-backed GTT on discrete GPU (over PCIe)
    DiscreteGtt,
    /// Pre-allocated hardware framebuffer carveout on APU / iGPU
    ApuCarveoutVram,
    /// Dynamically allocated GTT on APU / iGPU (shared DDR4)
    ApuSharedGtt,
    /// CPU Host System Memory (DDR4 / DDR5)
    #[default]
    HostSystemRam,
    /// Scratch or disk-backed offload domain
    StorageBacked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDomainInfo {
    pub id: DomainId,
    pub device_id: DeviceId,
    pub kind: MemoryDomainKind,
    pub total_capacity_bytes: u64,
    pub is_cpu_visible: bool,
    pub is_cpu_coherent: bool,
    pub is_device_local: bool,
    pub theoretical_bandwidth_gbps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainBudget {
    pub domain_id: DomainId,
    pub total_bytes: u64,
    pub driver_budget_bytes: u64,
    pub currently_allocated_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferUsage {
    ModelWeights,
    KvCache,
    ActivationBuffer,
    StagingTransfer,
    GeneralCompute,
}
