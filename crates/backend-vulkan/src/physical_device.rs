use ash::vk;
use openmemgpu_core::device::{DeviceId, DeviceKind, GpuArchitecture};
use openmemgpu_core::error::CoreError;
use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PciBusInfo {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
    pub pci_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanMemoryHeapInfo {
    pub index: usize,
    pub size_bytes: u64,
    pub is_device_local: bool,
    pub driver_budget_bytes: Option<u64>,
    pub driver_usage_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanMemoryTypeInfo {
    pub index: usize,
    pub heap_index: usize,
    pub property_flags: u32,
    pub is_device_local: bool,
    pub is_host_visible: bool,
    pub is_host_coherent: bool,
    pub is_host_cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanQueueFamily {
    pub index: u32,
    pub count: u32,
    pub supports_graphics: bool,
    pub supports_compute: bool,
    pub supports_transfer: bool,
    pub timestamp_valid_bits: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalDeviceDetails {
    pub id: DeviceId,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub kind: DeviceKind,
    pub architecture: GpuArchitecture,
    pub api_version: String,
    pub driver_version: String,
    pub device_uuid: [u8; 16],
    pub driver_uuid: [u8; 16],
    pub pci_bus: Option<PciBusInfo>,
    pub drm_card_node: Option<String>,
    pub drm_render_node: Option<String>,
    pub subgroup_size: u32,
    pub heaps: Vec<VulkanMemoryHeapInfo>,
    pub types: Vec<VulkanMemoryTypeInfo>,
    pub queue_families: Vec<VulkanQueueFamily>,
    pub dedicated_vram_bytes: u64,
}

impl PhysicalDeviceDetails {
    pub fn is_discrete(&self) -> bool {
        self.kind == DeviceKind::DiscreteGpu
    }

    pub fn is_integrated(&self) -> bool {
        self.kind == DeviceKind::IntegratedGpu
    }

    /// Select optimal memory type for allocation given memory requirements and desired property flags
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        desired_flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let flags_bits = desired_flags.as_raw();
        for (idx, mem_type) in self.types.iter().enumerate() {
            if (type_filter & (1 << idx)) != 0
                && (mem_type.property_flags & flags_bits) == flags_bits
            {
                return Some(idx as u32);
            }
        }
        None
    }
}

pub struct PhysicalDeviceEnumerator;

impl PhysicalDeviceEnumerator {
    pub fn enumerate(
        instance: &ash::Instance,
    ) -> Result<Vec<(vk::PhysicalDevice, PhysicalDeviceDetails)>, CoreError> {
        let pdevices = unsafe {
            instance.enumerate_physical_devices().map_err(|e| {
                CoreError::BackendInitFailed(format!(
                    "Failed to enumerate physical devices: {:?}",
                    e
                ))
            })?
        };

        let mut results = Vec::new();

        for (idx, &pdev) in pdevices.iter().enumerate() {
            let mut id_props = vk::PhysicalDeviceIDProperties::default();
            let mut pci_bus_props = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
            let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();

            let mut props2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut id_props)
                .push_next(&mut pci_bus_props)
                .push_next(&mut subgroup_props);

            unsafe {
                instance.get_physical_device_properties2(pdev, &mut props2);
            }

            let base_props = props2.properties;
            let name = unsafe { CStr::from_ptr(base_props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();

            let pci_info = if pci_bus_props.pci_bus != 0
                || pci_bus_props.pci_device != 0
                || pci_bus_props.pci_domain != 0
            {
                let addr = format!(
                    "{:04x}:{:02x}:{:02x}.{:x}",
                    pci_bus_props.pci_domain,
                    pci_bus_props.pci_bus,
                    pci_bus_props.pci_device,
                    pci_bus_props.pci_function
                );
                Some(PciBusInfo {
                    domain: pci_bus_props.pci_domain,
                    bus: pci_bus_props.pci_bus,
                    device: pci_bus_props.pci_device,
                    function: pci_bus_props.pci_function,
                    pci_address: addr,
                })
            } else {
                None
            };

            // Correlate with Linux DRM nodes
            let (drm_card, drm_render) =
                Self::correlate_drm(&pci_info, base_props.vendor_id, base_props.device_id);

            // Query memory properties and budget (VK_EXT_memory_budget)
            let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut mem_props2 =
                vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_props);

            unsafe {
                instance.get_physical_device_memory_properties2(pdev, &mut mem_props2);
            }

            let mem_props = mem_props2.memory_properties;
            let mut heaps = Vec::new();
            let mut dedicated_vram = 0u64;

            for h_idx in 0..(mem_props.memory_heap_count as usize) {
                let heap = mem_props.memory_heaps[h_idx];
                let is_dev_local = heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
                let budget = budget_props.heap_budget[h_idx];
                let usage = budget_props.heap_usage[h_idx];

                if is_dev_local && base_props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    dedicated_vram = heap.size;
                }

                heaps.push(VulkanMemoryHeapInfo {
                    index: h_idx,
                    size_bytes: heap.size,
                    is_device_local: is_dev_local,
                    driver_budget_bytes: if budget > 0 { Some(budget) } else { None },
                    driver_usage_bytes: Some(usage),
                });
            }

            let mut types = Vec::new();
            for t_idx in 0..(mem_props.memory_type_count as usize) {
                let mt = mem_props.memory_types[t_idx];
                types.push(VulkanMemoryTypeInfo {
                    index: t_idx,
                    heap_index: mt.heap_index as usize,
                    property_flags: mt.property_flags.as_raw(),
                    is_device_local: mt
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                    is_host_visible: mt
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                    is_host_coherent: mt
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_COHERENT),
                    is_host_cached: mt
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_CACHED),
                });
            }

            // Query Queue Families
            let qf_props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
            let queue_families: Vec<VulkanQueueFamily> = qf_props
                .iter()
                .enumerate()
                .map(|(q_idx, q)| VulkanQueueFamily {
                    index: q_idx as u32,
                    count: q.queue_count,
                    supports_graphics: q.queue_flags.contains(vk::QueueFlags::GRAPHICS),
                    supports_compute: q.queue_flags.contains(vk::QueueFlags::COMPUTE),
                    supports_transfer: q.queue_flags.contains(vk::QueueFlags::TRANSFER),
                    timestamp_valid_bits: q.timestamp_valid_bits,
                })
                .collect();

            let kind = match base_props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::DiscreteGpu,
                vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::IntegratedGpu,
                vk::PhysicalDeviceType::CPU => DeviceKind::CpuCompute,
                _ => DeviceKind::VirtualDevice,
            };

            let architecture = if base_props.vendor_id == 0x1002 {
                // AMD Vendor
                if base_props.device_id == 0x731f {
                    GpuArchitecture::Rdna1
                } else if base_props.device_id == 0x1636 {
                    GpuArchitecture::VegaGcn5
                } else {
                    GpuArchitecture::GenericVulkan
                }
            } else {
                GpuArchitecture::GenericVulkan
            };

            let api_ver = format!(
                "{}.{}.{}",
                vk::api_version_major(base_props.api_version),
                vk::api_version_minor(base_props.api_version),
                vk::api_version_patch(base_props.api_version)
            );

            let driver_ver = format!(
                "{}.{}.{}",
                vk::api_version_major(base_props.driver_version),
                vk::api_version_minor(base_props.driver_version),
                vk::api_version_patch(base_props.driver_version)
            );

            let details = PhysicalDeviceDetails {
                id: DeviceId(idx as u32),
                name,
                vendor_id: base_props.vendor_id,
                device_id: base_props.device_id,
                kind,
                architecture,
                api_version: api_ver,
                driver_version: driver_ver,
                device_uuid: id_props.device_uuid,
                driver_uuid: id_props.driver_uuid,
                pci_bus: pci_info,
                drm_card_node: drm_card,
                drm_render_node: drm_render,
                subgroup_size: subgroup_props.subgroup_size,
                heaps,
                types,
                queue_families,
                dedicated_vram_bytes: dedicated_vram,
            };

            results.push((pdev, details));
        }

        Ok(results)
    }

    fn correlate_drm(
        pci_info: &Option<PciBusInfo>,
        _vendor_id: u32,
        _device_id: u32,
    ) -> (Option<String>, Option<String>) {
        let pci_addr = match pci_info {
            Some(ref info) => &info.pci_address,
            None => return (None, None),
        };

        let mut card_node = None;
        let mut render_node = None;

        let drm_dir = Path::new("/sys/class/drm");
        if let Ok(entries) = fs::read_dir(drm_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let device_symlink = entry.path().join("device");
                if let Ok(target) = fs::read_link(&device_symlink) {
                    let target_str = target.to_string_lossy();
                    if target_str.contains(pci_addr) {
                        if name.starts_with("card") && !name.contains('-') {
                            card_node = Some(format!("/dev/dri/{}", name));
                        } else if name.starts_with("renderD") {
                            render_node = Some(format!("/dev/dri/{}", name));
                        }
                    }
                }
            }
        }

        (card_node, render_node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::VulkanContext;

    #[test]
    fn test_physical_device_enumeration() {
        let ctx = VulkanContext::new().expect("VulkanContext creation should succeed");
        let devices = PhysicalDeviceEnumerator::enumerate(&ctx.instance)
            .expect("Physical device enumeration should succeed");

        assert!(
            devices.len() >= 2,
            "Expected at least 2 physical devices (dGPU + iGPU), found {}",
            devices.len()
        );

        let mut found_dgpu = false;
        let mut found_igpu = false;

        for (_, dev) in &devices {
            println!(
                "Enumerated: {} ({:?}) - PCI: {:?}",
                dev.name, dev.kind, dev.pci_bus
            );
            println!("  DRM Render: {:?}", dev.drm_render_node);
            println!("  Subgroup Size: {} threads", dev.subgroup_size);
            println!("  Heaps count: {}", dev.heaps.len());
            for h in &dev.heaps {
                println!(
                    "    Heap {}: size={} MB, dev_local={}, budget={:?} MB, usage={:?} MB",
                    h.index,
                    h.size_bytes / 1024 / 1024,
                    h.is_device_local,
                    h.driver_budget_bytes.map(|b| b / 1024 / 1024),
                    h.driver_usage_bytes.map(|u| u / 1024 / 1024)
                );
            }

            if dev.is_discrete() {
                found_dgpu = true;
                assert!(
                    dev.subgroup_size == 32 || dev.subgroup_size == 64,
                    "Navi 10 subgroup size should be 32 or 64 (reported {})",
                    dev.subgroup_size
                );
                assert!(
                    dev.drm_render_node.is_some(),
                    "dGPU should correlate to DRM render node"
                );
            }
            if dev.is_integrated() {
                found_igpu = true;
                assert!(
                    dev.subgroup_size == 64,
                    "Vega 6 subgroup size must be 64 (Wave64)"
                );
                assert!(
                    dev.drm_render_node.is_some(),
                    "iGPU should correlate to DRM render node"
                );
            }
        }

        assert!(found_dgpu, "Should have enumerated Navi 10 dGPU");
        assert!(found_igpu, "Should have enumerated Renoir iGPU");
    }
}
