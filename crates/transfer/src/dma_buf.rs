use ash::vk;
use openmemgpu_backend_vulkan::context::VulkanContext;
use openmemgpu_backend_vulkan::physical_device::PhysicalDeviceEnumerator;
use openmemgpu_core::error::CoreError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceDmaBufCapabilities {
    pub device_name: String,
    pub pci_address: Option<String>,
    pub supports_external_memory_fd_extension: bool,
    pub dma_buf_exportable: bool,
    pub dma_buf_importable: bool,
    pub compatible_handle_types: u32,
    pub findings: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DmaBufFeasibilityReport {
    pub devices: Vec<DeviceDmaBufCapabilities>,
    pub direct_cross_gpu_dma_supported: bool,
    pub architectural_summary: String,
}

pub struct DmaBufProbe;

impl DmaBufProbe {
    pub fn probe() -> Result<DmaBufFeasibilityReport, CoreError> {
        let ctx = VulkanContext::new()?;
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)?;

        let mut device_caps = Vec::new();

        for (pdev, details) in &enumerated {
            // Check if VK_KHR_external_memory_fd is supported among device extension properties
            let dev_extensions = unsafe {
                ctx.instance
                    .enumerate_device_extension_properties(*pdev)
                    .unwrap_or_default()
            };

            let supports_ext_fd = dev_extensions.iter().any(|ext| {
                let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
                name.to_bytes() == b"VK_KHR_external_memory_fd"
            });

            let external_info = vk::PhysicalDeviceExternalBufferInfo::default()
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

            let mut external_props = vk::ExternalBufferProperties::default();

            unsafe {
                ctx.instance.get_physical_device_external_buffer_properties(
                    *pdev,
                    &external_info,
                    &mut external_props,
                );
            }

            let mem_props = external_props.external_memory_properties;
            let exportable = mem_props
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE);
            let importable = mem_props
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
            let compatible = mem_props.compatible_handle_types.as_raw();

            let findings = format!(
                "RADV Driver supports ext_fd: {}, DMA-BUF export: {}, import: {}, compatible_handles: {:#x}",
                supports_ext_fd, exportable, importable, compatible
            );

            device_caps.push(DeviceDmaBufCapabilities {
                device_name: details.name.clone(),
                pci_address: details.pci_bus.as_ref().map(|p| p.pci_address.clone()),
                supports_external_memory_fd_extension: supports_ext_fd,
                dma_buf_exportable: exportable,
                dma_buf_importable: importable,
                compatible_handle_types: compatible,
                findings,
            });
        }

        let navi10_export = device_caps
            .iter()
            .find(|d| d.device_name.contains("5600M") || d.device_name.contains("NAVI10"))
            .map(|d| d.dma_buf_exportable)
            .unwrap_or(false);

        let renoir_import = device_caps
            .iter()
            .find(|d| d.device_name.contains("RENOIR") || d.device_name.contains("Vega"))
            .map(|d| d.dma_buf_importable)
            .unwrap_or(false);

        let direct_supported = navi10_export && renoir_import;

        let architectural_summary = if direct_supported {
            "Hardware and RADV driver support VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT on both devices. Direct zero-copy or DRM PRIME export/import is theoretically feasible, subject to kernel amdgpu DMA-BUF memory-type pinning rules.".to_string()
        } else {
            "Direct cross-device DMA-BUF import/export is restricted by driver/subsystem. Host-mediated staging copy remains the primary reliable data movement path across discrete and integrated AMD GPU domains on Linux.".to_string()
        };

        Ok(DmaBufFeasibilityReport {
            devices: device_caps,
            direct_cross_gpu_dma_supported: direct_supported,
            architectural_summary,
        })
    }
}
