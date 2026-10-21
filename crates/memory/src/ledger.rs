use openmemgpu_core::error::CoreError;
use openmemgpu_core::memory::{DomainId, MemoryDomainKind};
use parking_lot::RwLock;
use std::collections::HashMap;

/// Mathematical model of physical memory domains.
/// Prevents double-counting shared backing stores (e.g. DDR4 host RAM shared by APU and host).
#[derive(Debug)]
pub struct PhysicalMemoryLedger {
    /// True physical dedicated VRAM capacity across discrete GPUs (bytes)
    dedicated_vram_capacity: u64,
    /// Currently allocated bytes in discrete VRAM
    dedicated_vram_allocated: RwLock<u64>,

    /// Total system physical RAM available to the OS (bytes)
    host_physical_ram_capacity: u64,
    /// Reserved headroom for operating system and user processes (bytes)
    os_headroom_reservation: u64,

    /// Live tracking of host-backed physical allocations across all consumers:
    /// (APU allocations + discrete GTT allocations + direct host allocations)
    host_physical_allocations: RwLock<HashMap<DomainId, DomainUsage>>,
}

#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
struct DomainUsage {
    kind: MemoryDomainKind,
    allocated_bytes: u64,
}

impl PhysicalMemoryLedger {
    pub fn new(
        dedicated_vram_capacity: u64,
        host_physical_ram_capacity: u64,
        os_headroom_reservation: u64,
    ) -> Self {
        Self {
            dedicated_vram_capacity,
            dedicated_vram_allocated: RwLock::new(0),
            host_physical_ram_capacity,
            os_headroom_reservation,
            host_physical_allocations: RwLock::new(HashMap::new()),
        }
    }

    /// Register an allocation against physical memory.
    /// Returns Err(OutOfMemory) if the physical budget is exceeded.
    pub fn reserve(
        &self,
        domain_id: DomainId,
        kind: MemoryDomainKind,
        size_bytes: u64,
    ) -> Result<(), CoreError> {
        match kind {
            MemoryDomainKind::DiscreteDedicatedVram => {
                let mut vram = self.dedicated_vram_allocated.write();
                let new_total = *vram + size_bytes;
                if new_total > self.dedicated_vram_capacity {
                    return Err(CoreError::OutOfMemory {
                        domain_id: format!("{:?}", domain_id),
                        requested_bytes: size_bytes,
                        available_bytes: self.dedicated_vram_capacity.saturating_sub(*vram),
                    });
                }
                *vram = new_total;
                Ok(())
            }
            MemoryDomainKind::ApuSharedGtt
            | MemoryDomainKind::ApuCarveoutVram
            | MemoryDomainKind::DiscreteGtt
            | MemoryDomainKind::HostSystemRam => {
                let mut allocations = self.host_physical_allocations.write();
                let current_total_host: u64 = allocations.values().map(|u| u.allocated_bytes).sum();
                let max_usable_host = self
                    .host_physical_ram_capacity
                    .saturating_sub(self.os_headroom_reservation);

                let new_total = current_total_host + size_bytes;
                if new_total > max_usable_host {
                    return Err(CoreError::OutOfMemory {
                        domain_id: format!("{:?} (Shared Host RAM)", domain_id),
                        requested_bytes: size_bytes,
                        available_bytes: max_usable_host.saturating_sub(current_total_host),
                    });
                }

                let entry = allocations.entry(domain_id).or_insert(DomainUsage {
                    kind,
                    allocated_bytes: 0,
                });
                entry.allocated_bytes += size_bytes;
                Ok(())
            }
            MemoryDomainKind::StorageBacked => Ok(()),
        }
    }

    /// Release previously reserved bytes.
    pub fn release(&self, domain_id: DomainId, kind: MemoryDomainKind, size_bytes: u64) {
        match kind {
            MemoryDomainKind::DiscreteDedicatedVram => {
                let mut vram = self.dedicated_vram_allocated.write();
                *vram = vram.saturating_sub(size_bytes);
            }
            MemoryDomainKind::ApuSharedGtt
            | MemoryDomainKind::ApuCarveoutVram
            | MemoryDomainKind::DiscreteGtt
            | MemoryDomainKind::HostSystemRam => {
                let mut allocations = self.host_physical_allocations.write();
                if let Some(entry) = allocations.get_mut(&domain_id) {
                    entry.allocated_bytes = entry.allocated_bytes.saturating_sub(size_bytes);
                }
            }
            MemoryDomainKind::StorageBacked => {}
        }
    }

    pub fn dedicated_vram_allocated(&self) -> u64 {
        *self.dedicated_vram_allocated.read()
    }

    pub fn total_host_ram_allocated(&self) -> u64 {
        self.host_physical_allocations
            .read()
            .values()
            .map(|u| u.allocated_bytes)
            .sum()
    }

    pub fn available_host_ram(&self) -> u64 {
        let max_usable = self
            .host_physical_ram_capacity
            .saturating_sub(self.os_headroom_reservation);
        max_usable.saturating_sub(self.total_host_ram_allocated())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_counting_prevention() {
        // Setup 6 GB dedicated VRAM, 14 GB System RAM, 2 GB OS reserve -> 12 GB usable RAM
        let ledger = PhysicalMemoryLedger::new(
            6 * 1024 * 1024 * 1024,
            14 * 1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
        );

        let apu_domain = DomainId(1);
        let gtt_domain = DomainId(2);

        // Reserve 8 GB on APU
        assert!(ledger
            .reserve(
                apu_domain,
                MemoryDomainKind::ApuSharedGtt,
                8 * 1024 * 1024 * 1024
            )
            .is_ok());

        // Attempting to reserve another 5 GB on discrete GTT should fail (8 + 5 = 13 > 12 GB max usable)
        let err = ledger.reserve(
            gtt_domain,
            MemoryDomainKind::DiscreteGtt,
            5 * 1024 * 1024 * 1024,
        );
        assert!(err.is_err());

        // Releasing 4 GB from APU makes room
        ledger.release(
            apu_domain,
            MemoryDomainKind::ApuSharedGtt,
            4 * 1024 * 1024 * 1024,
        );
        assert!(ledger
            .reserve(
                gtt_domain,
                MemoryDomainKind::DiscreteGtt,
                5 * 1024 * 1024 * 1024
            )
            .is_ok());
    }
}
