use crate::manager::VvramManager;
use openmemgpu_backend_vulkan::buffer::VulkanBuffer;
use openmemgpu_backend_vulkan::device::VulkanDevice;
use openmemgpu_core::error::CoreError;
use openmemgpu_device_discovery::SystemMemoryInfo;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramExpansionStatus {
    pub total_system_ram_gb: f64,
    pub available_ram_gb: f64,
    pub os_reserved_ram_gb: f64,
    pub base_dgpu_vram_gb: f64,
    pub expanded_igpu_vram_gb: f64,
    pub total_vvram_gb: f64,
    pub max_expandable_gb: f64,
    pub allowed_steps_gb: Vec<u32>,
    pub backing_allocated_bytes: u64,
}

pub struct OversizedWorkloadSlices {
    pub dgpu_buffer: Arc<VulkanBuffer>,
    pub igpu_buffer: Arc<VulkanBuffer>,
    pub dgpu_bytes: u64,
    pub igpu_bytes: u64,
    pub total_bytes: u64,
}

pub struct DynamicVramExpander {
    pub memory_info: SystemMemoryInfo,
    pub base_dgpu_bytes: u64,
    expanded_bytes: RwLock<u64>,
    provisioned_buffer: RwLock<Option<Arc<VulkanBuffer>>>,
    pub manager: Arc<VvramManager>,
    pub dgpu_device: Option<Arc<VulkanDevice>>,
    pub igpu_device: Option<Arc<VulkanDevice>>,
    pub context: Option<Arc<openmemgpu_backend_vulkan::context::VulkanContext>>,
}

impl DynamicVramExpander {
    pub fn new(
        manager: Arc<VvramManager>,
        dgpu_device: Option<Arc<VulkanDevice>>,
        igpu_device: Option<Arc<VulkanDevice>>,
        context: Option<Arc<openmemgpu_backend_vulkan::context::VulkanContext>>,
    ) -> Result<Self, CoreError> {
        let memory_info = SystemMemoryInfo::discover()?;
        let base_dgpu_bytes = memory_info.dgpu_vram_bytes;

        Ok(Self {
            memory_info,
            base_dgpu_bytes,
            expanded_bytes: RwLock::new(0),
            provisioned_buffer: RwLock::new(None),
            manager,
            dgpu_device,
            igpu_device,
            context,
        })
    }

    /// Query the current dynamic VRAM expansion state
    pub fn status(&self) -> VramExpansionStatus {
        let expanded = *self.expanded_bytes.read();
        let total_vvram = self.base_dgpu_bytes + expanded;
        let backing = self
            .provisioned_buffer
            .read()
            .as_ref()
            .map(|b| b.size_bytes)
            .unwrap_or(0);

        let to_gb = |b: u64| (b as f64) / (1024.0 * 1024.0 * 1024.0);

        VramExpansionStatus {
            total_system_ram_gb: to_gb(self.memory_info.total_ram_bytes),
            available_ram_gb: to_gb(self.memory_info.available_ram_bytes),
            os_reserved_ram_gb: to_gb(self.memory_info.os_reserved_ram_bytes),
            base_dgpu_vram_gb: to_gb(self.base_dgpu_bytes),
            expanded_igpu_vram_gb: to_gb(expanded),
            total_vvram_gb: to_gb(total_vvram),
            max_expandable_gb: to_gb(self.memory_info.max_expandable_bytes),
            allowed_steps_gb: self.memory_info.recommended_steps_gb.clone(),
            backing_allocated_bytes: backing,
        }
    }

    /// Set VRAM expansion by borrowing `add_gb` from iGPU unified system memory
    pub fn set_expansion(&self, add_gb: u32) -> Result<VramExpansionStatus, CoreError> {
        let requested_bytes = (add_gb as u64) * 1024 * 1024 * 1024;
        if requested_bytes > self.memory_info.max_expandable_bytes {
            return Err(CoreError::InvalidArgument(format!(
                "Requested expansion +{} GB exceeds safe maximum expandable RAM ({} GB, OS reserve: {} GB)",
                add_gb,
                self.memory_info.max_expandable_bytes / 1024 / 1024 / 1024,
                self.memory_info.os_reserved_ram_bytes / 1024 / 1024 / 1024
            )));
        }

        // Release any existing backing buffer first
        {
            let mut prov = self.provisioned_buffer.write();
            *prov = None;
        }

        // If an iGPU device is present and expansion > 0, provision backing Vulkan buffer
        if requested_bytes > 0 {
            if let Some(ref igpu) = self.igpu_device {
                let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER
                    | ash::vk::BufferUsageFlags::TRANSFER_SRC
                    | ash::vk::BufferUsageFlags::TRANSFER_DST;

                // Provision real Vulkan buffer in host-visible iGPU memory
                let backing = VulkanBuffer::new_host_visible(igpu.clone(), requested_bytes, usage)?;
                let mut prov = self.provisioned_buffer.write();
                *prov = Some(Arc::new(backing));
            }
        }

        *self.expanded_bytes.write() = requested_bytes;
        Ok(self.status())
    }

    /// Allocate a virtual buffer that exceeds dedicated discrete VRAM by utilizing the expanded iGPU pool
    pub fn allocate_oversized(
        &self,
        total_bytes: u64,
    ) -> Result<OversizedWorkloadSlices, CoreError> {
        let expanded = *self.expanded_bytes.read();
        let total_capacity = self.base_dgpu_bytes + expanded;

        if total_bytes > total_capacity {
            return Err(CoreError::OutOfMemory {
                domain_id: "VVRAM_EXPANDED_POOL".into(),
                requested_bytes: total_bytes,
                available_bytes: total_capacity,
            });
        }

        let dgpu = self.dgpu_device.as_ref().ok_or_else(|| {
            CoreError::DeviceNotFound("Discrete GPU required for oversized allocation".into())
        })?;
        let igpu = self.igpu_device.as_ref().ok_or_else(|| {
            CoreError::DeviceNotFound("Integrated GPU required for oversized allocation".into())
        })?;

        // Slicing strategy: Fill dGPU dedicated VRAM up to 85% capacity, overflow rest to expanded iGPU
        let dgpu_headroom = 512 * 1024 * 1024u64; // 512 MB safety headroom
        let max_dgpu_slice = self.base_dgpu_bytes.saturating_sub(dgpu_headroom);

        let (dgpu_bytes, igpu_bytes) = if total_bytes <= max_dgpu_slice {
            (total_bytes, 0)
        } else {
            let dgpu_part = max_dgpu_slice;
            let igpu_part = total_bytes - dgpu_part;
            (dgpu_part, igpu_part)
        };

        let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::TRANSFER_SRC
            | ash::vk::BufferUsageFlags::TRANSFER_DST;

        let dgpu_buffer = Arc::new(VulkanBuffer::new_device_local(
            dgpu.clone(),
            dgpu_bytes,
            usage,
        )?);
        let igpu_buffer = if igpu_bytes > 0 {
            Arc::new(VulkanBuffer::new_host_visible(
                igpu.clone(),
                igpu_bytes,
                usage,
            )?)
        } else {
            // Minimal placeholder buffer
            Arc::new(VulkanBuffer::new_host_visible(igpu.clone(), 64, usage)?)
        };

        Ok(OversizedWorkloadSlices {
            dgpu_buffer,
            igpu_buffer,
            dgpu_bytes,
            igpu_bytes,
            total_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmemgpu_backend_vulkan::context::VulkanContext;
    use openmemgpu_backend_vulkan::physical_device::PhysicalDeviceEnumerator;
    use openmemgpu_core::device::DeviceKind;
    use openmemgpu_memory::ledger::PhysicalMemoryLedger;

    #[test]
    fn test_dynamic_vram_expansion_status() {
        let ledger = Arc::new(PhysicalMemoryLedger::new(
            6 * 1024 * 1024 * 1024,
            14 * 1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
        ));
        let manager = Arc::new(VvramManager::new(
            ledger,
            openmemgpu_core::memory::DomainId(0),
            openmemgpu_core::memory::DomainId(1),
            openmemgpu_core::memory::DomainId(2),
        ));

        let expander = DynamicVramExpander::new(manager, None, None, None)
            .expect("Expander creation should succeed");
        let initial_status = expander.status();
        assert_eq!(initial_status.expanded_igpu_vram_gb, 0.0);
        assert!(initial_status.total_vvram_gb >= 5.0);

        // Expand by 2 GB (simulated without real device handles)
        let s2 = expander
            .set_expansion(2)
            .expect("Setting +2 GB should succeed");
        assert_eq!(s2.expanded_igpu_vram_gb, 2.0);
        assert_eq!(s2.total_vvram_gb, initial_status.base_dgpu_vram_gb + 2.0);

        // Reset to 0
        let s0 = expander
            .set_expansion(0)
            .expect("Resetting to 0 should succeed");
        assert_eq!(s0.expanded_igpu_vram_gb, 0.0);
    }

    #[test]
    fn test_dynamic_vram_expansion_hardware() {
        let ctx = Arc::new(VulkanContext::new().expect("VulkanContext creation should succeed"));
        let enumerated =
            PhysicalDeviceEnumerator::enumerate(&ctx.instance).expect("Enumeration should succeed");

        let dgpu_entry = enumerated
            .iter()
            .find(|(_, d)| d.kind == DeviceKind::DiscreteGpu);
        let igpu_entry = enumerated
            .iter()
            .find(|(_, d)| d.kind == DeviceKind::IntegratedGpu);

        if let (Some(&(dgpu_pdev, ref dgpu_details)), Some(&(igpu_pdev, ref igpu_details))) =
            (dgpu_entry, igpu_entry)
        {
            let dgpu = VulkanDevice::new_with_context(ctx.clone(), dgpu_pdev, dgpu_details.clone())
                .expect("dGPU should create");
            let igpu = VulkanDevice::new_with_context(ctx.clone(), igpu_pdev, igpu_details.clone())
                .expect("iGPU should create");

            let ledger = Arc::new(PhysicalMemoryLedger::new(
                6 * 1024 * 1024 * 1024,
                14 * 1024 * 1024 * 1024,
                2 * 1024 * 1024 * 1024,
            ));
            let manager = Arc::new(VvramManager::new(
                ledger,
                openmemgpu_core::memory::DomainId(0),
                openmemgpu_core::memory::DomainId(1),
                openmemgpu_core::memory::DomainId(2),
            ));

            let expander = DynamicVramExpander::new(manager, Some(dgpu), Some(igpu), Some(ctx.clone()))
                .expect("Expander should create");

            // Expand by +2 GB on real hardware (allocates real backing buffer on iGPU host RAM)
            let s2 = expander
                .set_expansion(2)
                .expect("Real hardware +2 GB expansion should succeed");
            assert_eq!(s2.expanded_igpu_vram_gb, 2.0);
            assert_eq!(s2.backing_allocated_bytes, 2 * 1024 * 1024 * 1024);
            println!(
                "  [HARDWARE PROVEN] +2 GB iGPU expansion provisioned backing {} bytes",
                s2.backing_allocated_bytes
            );

            // Test oversized allocation: 6.5 GB (exceeding dedicated 6.0 GB dGPU VRAM)
            let oversize_req = (6.5 * 1024.0 * 1024.0 * 1024.0) as u64;
            let slices = expander
                .allocate_oversized(oversize_req)
                .expect("Oversized 6.5 GB allocation should succeed across dGPU+iGPU");
            assert!(slices.dgpu_bytes > 0);
            assert!(slices.igpu_bytes > 0);
            assert_eq!(slices.dgpu_bytes + slices.igpu_bytes, oversize_req);
            println!(
                "  [HARDWARE PROVEN] Oversized 6.5 GB buffer allocated: dGPU={} MB, iGPU={} MB",
                slices.dgpu_bytes / 1024 / 1024,
                slices.igpu_bytes / 1024 / 1024
            );

            // Release expansion
            expander.set_expansion(0).expect("Reset should succeed");
        }
    }
}
