use openmemgpu_core::memory::DomainId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyState {
    /// Logical descriptor registered, physical allocation not yet materialized
    AllocatedLogical,
    /// Physically resident in discrete GPU dedicated VRAM
    ResidentDiscreteVram(DomainId),
    /// Physically resident in integrated GPU / APU shared memory
    ResidentApuMemory(DomainId),
    /// Resident in Host CPU system RAM (coherent staging or eviction backing)
    ResidentHostRam(DomainId),
    /// In transit between memory domains via asynchronous DMA copy
    InTransit {
        source: DomainId,
        destination: DomainId,
    },
    /// Replicated across multiple memory domains (e.g. read-only weights)
    Replicated,
    /// Evicted to system RAM or storage under memory pressure
    Evicted,
    /// Released and unmapped
    Released,
}

impl ResidencyState {
    pub fn is_resident(&self) -> bool {
        matches!(
            self,
            Self::ResidentDiscreteVram(_)
                | Self::ResidentApuMemory(_)
                | Self::ResidentHostRam(_)
                | Self::Replicated
        )
    }

    pub fn can_transition_to(&self, target: &ResidencyState) -> bool {
        matches!(
            (self, target),
            (Self::AllocatedLogical, Self::ResidentDiscreteVram(_))
                | (Self::AllocatedLogical, Self::ResidentApuMemory(_))
                | (Self::AllocatedLogical, Self::ResidentHostRam(_))
                | (Self::ResidentDiscreteVram(_), Self::InTransit { .. })
                | (Self::ResidentApuMemory(_), Self::InTransit { .. })
                | (Self::ResidentHostRam(_), Self::InTransit { .. })
                | (Self::InTransit { .. }, Self::ResidentDiscreteVram(_))
                | (Self::InTransit { .. }, Self::ResidentApuMemory(_))
                | (Self::InTransit { .. }, Self::ResidentHostRam(_))
                | (Self::ResidentDiscreteVram(_), Self::Evicted)
                | (Self::ResidentApuMemory(_), Self::Evicted)
                | (Self::Evicted, Self::ResidentHostRam(_))
                | (_, Self::Released)
        )
    }
}
