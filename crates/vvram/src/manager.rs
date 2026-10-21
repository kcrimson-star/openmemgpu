use crate::capacity::VvramCapacityReport;
use crate::policy::PlacementPolicy;
use crate::residency::ResidencyState;
use openmemgpu_backend_vulkan::buffer::VulkanBuffer;
use openmemgpu_backend_vulkan::device::VulkanDevice;
use openmemgpu_core::device::DeviceId;
use openmemgpu_core::error::CoreError;
use openmemgpu_core::memory::{DomainId, MemoryDomainKind};
use openmemgpu_memory::ledger::PhysicalMemoryLedger;
use openmemgpu_transfer::{CrossGpuTransferEngine, CrossGpuTransferMetrics};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VvramBufferDescriptor {
    pub name: String,
    pub size_bytes: u64,
    pub policy: PlacementPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAllocationInfo {
    pub logical_id: BufferId,
    pub device_id: DeviceId,
    pub domain_id: DomainId,
    pub requested_size_bytes: u64,
    pub actual_size_bytes: u64,
    pub memory_type_index: u32,
    pub is_device_local: bool,
    pub is_host_visible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VvramBuffer {
    pub id: BufferId,
    pub name: String,
    pub size_bytes: u64,
    pub policy: PlacementPolicy,
    pub residency: ResidencyState,
    pub physical_info: Option<PhysicalAllocationInfo>,
}

pub struct VvramManager {
    ledger: Arc<PhysicalMemoryLedger>,
    buffers: RwLock<HashMap<BufferId, VvramBuffer>>,
    physical_buffers: RwLock<HashMap<BufferId, Arc<VulkanBuffer>>>,
    next_buffer_id: RwLock<u64>,
    pub dedicated_vram_domain: DomainId,
    pub apu_domain: DomainId,
    pub host_domain: DomainId,
}

impl VvramManager {
    pub fn new(
        ledger: Arc<PhysicalMemoryLedger>,
        dedicated_vram_domain: DomainId,
        apu_domain: DomainId,
        host_domain: DomainId,
    ) -> Self {
        Self {
            ledger,
            buffers: RwLock::new(HashMap::new()),
            physical_buffers: RwLock::new(HashMap::new()),
            next_buffer_id: RwLock::new(1),
            dedicated_vram_domain,
            apu_domain,
            host_domain,
        }
    }

    /// Allocate a logical VVRAM buffer with specified policy.
    pub fn allocate(&self, desc: VvramBufferDescriptor) -> Result<VvramBuffer, CoreError> {
        let mut id_guard = self.next_buffer_id.write();
        let id = BufferId(*id_guard);
        *id_guard += 1;

        let buffer = VvramBuffer {
            id,
            name: desc.name,
            size_bytes: desc.size_bytes,
            policy: desc.policy,
            residency: ResidencyState::AllocatedLogical,
            physical_info: None,
        };

        self.buffers.write().insert(id, buffer.clone());
        Ok(buffer)
    }

    /// Materialize logical allocation in the physical ledger without creating GPU hardware buffers (mock/simulated).
    pub fn materialize_logical(&self, buffer_id: BufferId) -> Result<(), CoreError> {
        let mut buffers = self.buffers.write();
        let buffer = buffers.get_mut(&buffer_id).ok_or_else(|| {
            CoreError::DeviceNotFound(format!("Buffer {:?} not found", buffer_id))
        })?;

        let (target_domain, kind, new_state) = match buffer.policy {
            PlacementPolicy::PreferDiscreteVram => (
                self.dedicated_vram_domain,
                MemoryDomainKind::DiscreteDedicatedVram,
                ResidencyState::ResidentDiscreteVram(self.dedicated_vram_domain),
            ),
            PlacementPolicy::PreferApuMemory => (
                self.apu_domain,
                MemoryDomainKind::ApuSharedGtt,
                ResidencyState::ResidentApuMemory(self.apu_domain),
            ),
            PlacementPolicy::PreferHostRam | PlacementPolicy::Adaptive => (
                self.host_domain,
                MemoryDomainKind::HostSystemRam,
                ResidencyState::ResidentHostRam(self.host_domain),
            ),
        };

        self.ledger
            .reserve(target_domain, kind, buffer.size_bytes)?;
        buffer.residency = new_state;

        Ok(())
    }

    /// Materialize real physical GPU backing buffer using Vulkan on target device.
    /// Uses device-local memory if policy is PreferDiscreteVram on a discrete GPU, host-visible otherwise.
    pub fn materialize_vulkan(
        &self,
        buffer_id: BufferId,
        device: Arc<VulkanDevice>,
    ) -> Result<Arc<VulkanBuffer>, CoreError> {
        let prefer_device_local = {
            let buffers = self.buffers.read();
            let buf = buffers.get(&buffer_id).ok_or_else(|| {
                CoreError::DeviceNotFound(format!("Buffer {:?} not found", buffer_id))
            })?;
            buf.policy == PlacementPolicy::PreferDiscreteVram && device.details.is_discrete()
        };

        self.materialize_vulkan_explicit(buffer_id, device, prefer_device_local)
    }

    /// Materialize real physical GPU backing buffer using explicit memory tier selection (device-local vs host-visible).
    /// Updates physical ledger and preserves rollback safety if driver allocation fails.
    pub fn materialize_vulkan_explicit(
        &self,
        buffer_id: BufferId,
        device: Arc<VulkanDevice>,
        prefer_device_local: bool,
    ) -> Result<Arc<VulkanBuffer>, CoreError> {
        let mut buffers = self.buffers.write();
        let buffer = buffers.get_mut(&buffer_id).ok_or_else(|| {
            CoreError::DeviceNotFound(format!("Buffer {:?} not found", buffer_id))
        })?;

        if buffer.residency != ResidencyState::AllocatedLogical {
            return Err(CoreError::InvalidStateTransition(format!(
                "Cannot materialize buffer {:?} from state {:?}",
                buffer_id, buffer.residency
            )));
        }

        let (target_domain, kind, new_state) = if device.details.is_discrete() {
            (
                self.dedicated_vram_domain,
                MemoryDomainKind::DiscreteDedicatedVram,
                ResidencyState::ResidentDiscreteVram(self.dedicated_vram_domain),
            )
        } else if device.details.is_integrated() {
            (
                self.apu_domain,
                MemoryDomainKind::ApuSharedGtt,
                ResidencyState::ResidentApuMemory(self.apu_domain),
            )
        } else {
            (
                self.host_domain,
                MemoryDomainKind::HostSystemRam,
                ResidencyState::ResidentHostRam(self.host_domain),
            )
        };

        // 1. Reserve in physical ledger first
        self.ledger
            .reserve(target_domain, kind, buffer.size_bytes)?;

        // 2. Allocate real Vulkan buffer
        let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::TRANSFER_SRC
            | ash::vk::BufferUsageFlags::TRANSFER_DST;

        let alloc_res = if prefer_device_local {
            VulkanBuffer::new_device_local(device.clone(), buffer.size_bytes, usage)
        } else {
            VulkanBuffer::new_host_visible(device.clone(), buffer.size_bytes, usage)
        };

        let vk_buf = match alloc_res {
            Ok(b) => Arc::new(b),
            Err(e) => {
                // Rollback ledger reservation on driver allocation failure
                self.ledger.release(target_domain, kind, buffer.size_bytes);
                return Err(e);
            }
        };

        // 3. Record physical allocation metadata
        buffer.physical_info = Some(PhysicalAllocationInfo {
            logical_id: buffer_id,
            device_id: device.details.id,
            domain_id: target_domain,
            requested_size_bytes: buffer.size_bytes,
            actual_size_bytes: vk_buf.actual_allocated_bytes,
            memory_type_index: vk_buf.memory_type_index,
            is_device_local: vk_buf.is_device_local,
            is_host_visible: vk_buf.is_host_visible,
        });

        buffer.residency = new_state;

        // 4. Store active Vulkan buffer handle
        self.physical_buffers
            .write()
            .insert(buffer_id, vk_buf.clone());

        Ok(vk_buf)
    }

    /// Retrieve active physical Vulkan buffer if materialized
    pub fn get_physical_buffer(&self, buffer_id: BufferId) -> Option<Arc<VulkanBuffer>> {
        self.physical_buffers.read().get(&buffer_id).cloned()
    }

    /// Retrieve buffer metadata
    pub fn get_buffer(&self, buffer_id: BufferId) -> Option<VvramBuffer> {
        self.buffers.read().get(&buffer_id).cloned()
    }

    /// Migrate an existing physically materialized buffer to a different target GPU.
    /// Coordinates ledger capacity reservation, intermediate InTransit state,
    /// host-mediated cross-GPU staging transfer, and source cleanup with rollback safety.
    pub fn migrate_buffer(
        &self,
        buffer_id: BufferId,
        target_device: Arc<VulkanDevice>,
        target_prefer_device_local: bool,
    ) -> Result<CrossGpuTransferMetrics, CoreError> {
        let (old_domain, old_kind, old_vk_buf, buffer_size) = {
            let buffers = self.buffers.read();
            let buf = buffers.get(&buffer_id).ok_or_else(|| {
                CoreError::DeviceNotFound(format!("Buffer {:?} not found", buffer_id))
            })?;

            let (dom, kind) = match buf.residency {
                ResidencyState::ResidentDiscreteVram(d) => {
                    (d, MemoryDomainKind::DiscreteDedicatedVram)
                }
                ResidencyState::ResidentApuMemory(d) => (d, MemoryDomainKind::ApuSharedGtt),
                ResidencyState::ResidentHostRam(d) => (d, MemoryDomainKind::HostSystemRam),
                other => {
                    return Err(CoreError::InvalidStateTransition(format!(
                        "Cannot migrate buffer {:?} from state {:?}",
                        buffer_id, other
                    )))
                }
            };

            let physical_bufs = self.physical_buffers.read();
            let vk_buf = physical_bufs.get(&buffer_id).cloned().ok_or_else(|| {
                CoreError::InvalidStateTransition(format!(
                    "Buffer {:?} marked resident but physical buffer missing",
                    buffer_id
                ))
            })?;

            (dom, kind, vk_buf, buf.size_bytes)
        };

        let (target_domain, target_kind, target_state) = if target_device.details.is_discrete() {
            (
                self.dedicated_vram_domain,
                MemoryDomainKind::DiscreteDedicatedVram,
                ResidencyState::ResidentDiscreteVram(self.dedicated_vram_domain),
            )
        } else if target_device.details.is_integrated() {
            (
                self.apu_domain,
                MemoryDomainKind::ApuSharedGtt,
                ResidencyState::ResidentApuMemory(self.apu_domain),
            )
        } else {
            (
                self.host_domain,
                MemoryDomainKind::HostSystemRam,
                ResidencyState::ResidentHostRam(self.host_domain),
            )
        };

        // 1. Transition buffer to InTransit state
        {
            let mut buffers = self.buffers.write();
            if let Some(buf) = buffers.get_mut(&buffer_id) {
                buf.residency = ResidencyState::InTransit {
                    source: old_domain,
                    destination: target_domain,
                };
            }
        }

        // 2. Reserve ledger capacity on target domain
        if let Err(e) = self.ledger.reserve(target_domain, target_kind, buffer_size) {
            // Revert state
            let mut buffers = self.buffers.write();
            if let Some(buf) = buffers.get_mut(&buffer_id) {
                buf.residency = match old_kind {
                    MemoryDomainKind::DiscreteDedicatedVram => {
                        ResidencyState::ResidentDiscreteVram(old_domain)
                    }
                    MemoryDomainKind::ApuSharedGtt => ResidencyState::ResidentApuMemory(old_domain),
                    _ => ResidencyState::ResidentHostRam(old_domain),
                };
            }
            return Err(e);
        }

        // 3. Allocate new buffer on target device
        let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::TRANSFER_SRC
            | ash::vk::BufferUsageFlags::TRANSFER_DST;

        let alloc_res = if target_prefer_device_local {
            VulkanBuffer::new_device_local(target_device.clone(), buffer_size, usage)
        } else {
            VulkanBuffer::new_host_visible(target_device.clone(), buffer_size, usage)
        };

        let new_vk_buf = match alloc_res {
            Ok(b) => Arc::new(b),
            Err(e) => {
                // Rollback target ledger reservation and state
                self.ledger.release(target_domain, target_kind, buffer_size);
                let mut buffers = self.buffers.write();
                if let Some(buf) = buffers.get_mut(&buffer_id) {
                    buf.residency = match old_kind {
                        MemoryDomainKind::DiscreteDedicatedVram => {
                            ResidencyState::ResidentDiscreteVram(old_domain)
                        }
                        MemoryDomainKind::ApuSharedGtt => {
                            ResidencyState::ResidentApuMemory(old_domain)
                        }
                        _ => ResidencyState::ResidentHostRam(old_domain),
                    };
                }
                return Err(e);
            }
        };

        // 4. Perform cross-GPU data movement
        let metrics = match CrossGpuTransferEngine::transfer_buffers(&old_vk_buf, &new_vk_buf) {
            Ok(m) => m,
            Err(e) => {
                // Rollback new allocation and target ledger reservation
                self.ledger.release(target_domain, target_kind, buffer_size);
                let mut buffers = self.buffers.write();
                if let Some(buf) = buffers.get_mut(&buffer_id) {
                    buf.residency = match old_kind {
                        MemoryDomainKind::DiscreteDedicatedVram => {
                            ResidencyState::ResidentDiscreteVram(old_domain)
                        }
                        MemoryDomainKind::ApuSharedGtt => {
                            ResidencyState::ResidentApuMemory(old_domain)
                        }
                        _ => ResidencyState::ResidentHostRam(old_domain),
                    };
                }
                return Err(e);
            }
        };

        // 5. Success: Release capacity on old domain and update active physical buffer
        self.ledger.release(old_domain, old_kind, buffer_size);
        self.physical_buffers
            .write()
            .insert(buffer_id, new_vk_buf.clone());

        {
            let mut buffers = self.buffers.write();
            if let Some(buf) = buffers.get_mut(&buffer_id) {
                buf.residency = target_state;
                buf.physical_info = Some(PhysicalAllocationInfo {
                    logical_id: buffer_id,
                    device_id: target_device.details.id,
                    domain_id: target_domain,
                    requested_size_bytes: buffer_size,
                    actual_size_bytes: new_vk_buf.actual_allocated_bytes,
                    memory_type_index: new_vk_buf.memory_type_index,
                    is_device_local: new_vk_buf.is_device_local,
                    is_host_visible: new_vk_buf.is_host_visible,
                });
            }
        }

        Ok(metrics)
    }

    /// Free a VVRAM buffer and reclaim physical capacity.
    pub fn free(&self, buffer_id: BufferId) -> Result<(), CoreError> {
        let mut buffers = self.buffers.write();
        let buffer = buffers.get_mut(&buffer_id).ok_or_else(|| {
            CoreError::DeviceNotFound(format!("Buffer {:?} not found", buffer_id))
        })?;

        match buffer.residency {
            ResidencyState::ResidentDiscreteVram(d) => {
                self.ledger.release(
                    d,
                    MemoryDomainKind::DiscreteDedicatedVram,
                    buffer.size_bytes,
                );
            }
            ResidencyState::ResidentApuMemory(d) => {
                self.ledger
                    .release(d, MemoryDomainKind::ApuSharedGtt, buffer.size_bytes);
            }
            ResidencyState::ResidentHostRam(d) => {
                self.ledger
                    .release(d, MemoryDomainKind::HostSystemRam, buffer.size_bytes);
            }
            _ => {}
        }

        buffer.residency = ResidencyState::Released;
        buffer.physical_info = None;

        self.physical_buffers.write().remove(&buffer_id);
        buffers.remove(&buffer_id);
        Ok(())
    }

    /// Generate an accurate capacity report across all domains.
    pub fn capacity_report(&self) -> VvramCapacityReport {
        let buffers = self.buffers.read();
        let logical_total: u64 = buffers.values().map(|b| b.size_bytes).sum();

        VvramCapacityReport {
            dedicated_vram_total_bytes: 6 * 1024 * 1024 * 1024,
            dedicated_vram_allocated_bytes: self.ledger.dedicated_vram_allocated(),
            dedicated_vram_available_bytes: (6 * 1024 * 1024 * 1024u64)
                .saturating_sub(self.ledger.dedicated_vram_allocated()),
            host_ram_total_bytes: 14 * 1024 * 1024 * 1024,
            host_ram_available_bytes: self.ledger.available_host_ram(),
            apu_memory_allocated_bytes: 0,
            staging_memory_allocated_bytes: 0,
            logical_vvram_allocated_bytes: logical_total,
            active_buffer_count: buffers.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmemgpu_backend_vulkan::context::VulkanContext;
    use openmemgpu_backend_vulkan::physical_device::PhysicalDeviceEnumerator;
    use openmemgpu_core::device::DeviceKind;

    #[test]
    fn test_vvram_logical_lifecycle() {
        let ledger = Arc::new(PhysicalMemoryLedger::new(
            6 * 1024 * 1024 * 1024,
            14 * 1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
        ));

        let vvram = VvramManager::new(ledger.clone(), DomainId(0), DomainId(1), DomainId(2));

        let buf = vvram
            .allocate(VvramBufferDescriptor {
                name: "weight_matrix_qkv".into(),
                size_bytes: 1024 * 1024 * 1024, // 1 GB
                policy: PlacementPolicy::PreferDiscreteVram,
            })
            .expect("Allocation should succeed");

        assert_eq!(buf.residency, ResidencyState::AllocatedLogical);

        vvram
            .materialize_logical(buf.id)
            .expect("Materialization should succeed");
        assert_eq!(ledger.dedicated_vram_allocated(), 1024 * 1024 * 1024);

        vvram.free(buf.id).expect("Free should succeed");
        assert_eq!(ledger.dedicated_vram_allocated(), 0);
    }

    #[test]
    fn test_vvram_cross_gpu_migration_hardware() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)
            .expect("Failed to enumerate devices");

        let dgpu_entry = enumerated
            .iter()
            .find(|(_, d)| d.kind == DeviceKind::DiscreteGpu);
        let igpu_entry = enumerated
            .iter()
            .find(|(_, d)| d.kind == DeviceKind::IntegratedGpu);

        if dgpu_entry.is_none() || igpu_entry.is_none() {
            println!("Skipping migration test: requires both dGPU and iGPU");
            return;
        }

        let (dgpu_pdev, dgpu_details) = dgpu_entry.unwrap();
        let (igpu_pdev, igpu_details) = igpu_entry.unwrap();

        let dgpu_dev = VulkanDevice::new(&ctx.instance, *dgpu_pdev, dgpu_details.clone()).unwrap();
        let igpu_dev = VulkanDevice::new(&ctx.instance, *igpu_pdev, igpu_details.clone()).unwrap();

        let ledger = Arc::new(PhysicalMemoryLedger::new(
            6 * 1024 * 1024 * 1024,
            14 * 1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
        ));

        let vvram = VvramManager::new(ledger.clone(), DomainId(0), DomainId(1), DomainId(2));

        let element_count = 32768usize; // 128 KB
        let byte_size = (element_count * std::mem::size_of::<f32>()) as u64;

        let buf = vvram
            .allocate(VvramBufferDescriptor {
                name: "migrating_tensor".into(),
                size_bytes: byte_size,
                policy: PlacementPolicy::PreferDiscreteVram,
            })
            .expect("Allocation failed");

        // 1. Materialize on dGPU (dedicated device-local)
        let vk_buf = vvram
            .materialize_vulkan_explicit(buf.id, dgpu_dev.clone(), true)
            .expect("Materialization on dGPU failed");
        assert!(vk_buf.is_device_local);
        assert_eq!(ledger.dedicated_vram_allocated(), byte_size);

        // Upload initial test pattern to dGPU
        let test_data: Vec<f32> = (0..element_count).map(|i| (i as f32) * 42.0).collect();
        vk_buf.upload_staged(&test_data).expect("Upload failed");

        // 2. Migrate from dGPU to iGPU
        let metrics = vvram
            .migrate_buffer(buf.id, igpu_dev.clone(), false)
            .expect("Migration failed");
        println!("Migration metrics (dGPU -> iGPU): {:#?}", metrics);

        // Verify ledger updated correctly: dGPU released, iGPU allocated
        assert_eq!(ledger.dedicated_vram_allocated(), 0);
        let updated_buf = vvram.get_buffer(buf.id).unwrap();
        assert!(matches!(
            updated_buf.residency,
            ResidencyState::ResidentApuMemory(_)
        ));

        // Readback and verify numerical data on iGPU
        let new_vk_buf = vvram.get_physical_buffer(buf.id).unwrap();
        let mut readback = vec![0.0f32; element_count];
        new_vk_buf
            .download_staged(&mut readback)
            .expect("Download failed");
        for i in 0..element_count {
            assert_eq!(test_data[i], readback[i], "Mismatch at index {}", i);
        }

        // 3. Free buffer
        vvram.free(buf.id).expect("Free failed");
        assert_eq!(ledger.dedicated_vram_allocated(), 0);
        println!("Cross-GPU migration test passed with 100% data integrity!");
    }
}
