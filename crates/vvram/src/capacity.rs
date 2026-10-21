use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VvramCapacityReport {
    /// Dedicated discrete VRAM total physical capacity (bytes)
    pub dedicated_vram_total_bytes: u64,
    /// Dedicated discrete VRAM allocated bytes
    pub dedicated_vram_allocated_bytes: u64,
    /// Live remaining budget in discrete VRAM
    pub dedicated_vram_available_bytes: u64,

    /// Host system RAM total capacity (bytes)
    pub host_ram_total_bytes: u64,
    /// Host system RAM currently available (excluding OS reserve and active allocations)
    pub host_ram_available_bytes: u64,

    /// Shared memory currently allocated to iGPU / APU
    pub apu_memory_allocated_bytes: u64,
    /// Memory used by host staging buffers for transfers
    pub staging_memory_allocated_bytes: u64,

    /// Logical VVRAM virtual allocation total (bytes)
    pub logical_vvram_allocated_bytes: u64,
    /// Number of active VVRAM buffers
    pub active_buffer_count: usize,
}
