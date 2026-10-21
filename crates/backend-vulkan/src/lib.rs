pub mod buffer;
pub mod compute;
pub mod context;
pub mod device;
pub mod physical_device;
pub mod shader;

pub use buffer::VulkanBuffer;
pub use compute::{ComputePipeline, GemmPipeline, ScaleBiasPipeline, TwoStageComputePipeline};
pub use context::VulkanContext;
pub use device::VulkanDevice;
use openmemgpu_core::error::CoreError;
pub use physical_device::{PhysicalDeviceDetails, PhysicalDeviceEnumerator};
pub use shader::ShaderCompiler;

pub struct VulkanBackend {
    pub is_available: bool,
}

impl VulkanBackend {
    pub fn init() -> Result<Self, CoreError> {
        // Safe check for libvulkan.so.1 presence
        let path = std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so.1");
        if !path.exists() {
            return Err(CoreError::BackendInitFailed(
                "libvulkan.so.1 not found on system".into(),
            ));
        }

        Ok(Self { is_available: true })
    }
}
