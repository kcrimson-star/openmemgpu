use ash::vk;
use openmemgpu_core::error::CoreError;
use std::ffi::{CStr, CString};

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
}

impl VulkanContext {
    pub fn new() -> Result<Self, CoreError> {
        let entry = unsafe {
            ash::Entry::load().map_err(|e| {
                CoreError::BackendInitFailed(format!("Failed to load Vulkan entry point: {:?}", e))
            })?
        };

        let app_name = CString::new("OpenMemGPU").unwrap();
        let engine_name = CString::new("OpenMemGPU-Engine").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let available_extensions = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .map_err(|e| {
                    CoreError::BackendInitFailed(format!("Enumerate extensions failed: {:?}", e))
                })?
        };

        let mut extension_names: Vec<*const i8> = Vec::new();
        let mut has_props2 = false;

        for ext in &available_extensions {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            if name.to_bytes() == b"VK_KHR_get_physical_device_properties2" {
                has_props2 = true;
                extension_names.push(ext.extension_name.as_ptr());
            }
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        let instance = unsafe {
            entry.create_instance(&create_info, None).map_err(|e| {
                CoreError::BackendInitFailed(format!("Create instance failed: {:?}", e))
            })?
        };

        tracing::info!(
            "Vulkan instance created successfully (VK_KHR_get_physical_device_properties2: {})",
            has_props2
        );

        Ok(Self { entry, instance })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_instance_creation() {
        let ctx = VulkanContext::new();
        assert!(
            ctx.is_ok(),
            "Failed to create VulkanContext: {:?}",
            ctx.err()
        );
    }
}
