use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlacementPolicy {
    /// High-throughput compute: Strictly target dedicated VRAM on dGPU. Fallback to APU if full.
    PreferDiscreteVram,
    /// APU-primary: Place in APU shared memory (avoids PCIe transfer when iGPU computes).
    PreferApuMemory,
    /// Host-staging: Place directly in host-visible coherent system RAM.
    PreferHostRam,
    /// Dynamic: Evaluate live memory budgets and bandwidth requirements automatically.
    Adaptive,
}
