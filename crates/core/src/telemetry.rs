use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub timestamp_unix_ms: u64,
    pub total_dedicated_vram_allocated_bytes: u64,
    pub total_apu_memory_allocated_bytes: u64,
    pub total_host_memory_allocated_bytes: u64,
    pub active_vvram_buffers: usize,
    pub transfer_volume_bytes: u64,
}

pub trait TelemetrySink: Send + Sync {
    fn record(&self, snapshot: TelemetrySnapshot);
}
