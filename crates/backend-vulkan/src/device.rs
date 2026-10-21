use crate::physical_device::PhysicalDeviceDetails;
use ash::vk;
use openmemgpu_core::error::CoreError;
use std::sync::Arc;

pub struct VulkanDevice {
    pub raw: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub details: PhysicalDeviceDetails,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    pub non_coherent_atom_size: u64,
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    pub fn new(
        instance: &ash::Instance,
        pdev: vk::PhysicalDevice,
        details: PhysicalDeviceDetails,
    ) -> Result<Arc<Self>, CoreError> {
        let dev_props = unsafe { instance.get_physical_device_properties(pdev) };
        let non_coherent_atom_size = dev_props.limits.non_coherent_atom_size;

        let compute_family_index = details
            .queue_families
            .iter()
            .find(|q| q.supports_compute && !q.supports_graphics)
            .or_else(|| details.queue_families.iter().find(|q| q.supports_compute))
            .map(|q| q.index)
            .ok_or_else(|| CoreError::BackendInitFailed("No compute queue family found".into()))?;

        let queue_priorities = [1.0f32];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family_index)
            .queue_priorities(&queue_priorities)];

        let mut vulkan12_features =
            vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .push_next(&mut vulkan12_features);

        let device = unsafe {
            instance
                .create_device(pdev, &device_create_info, None)
                .map_err(|e| {
                    CoreError::BackendInitFailed(format!(
                        "Failed to create Vulkan logical device: {:?}",
                        e
                    ))
                })?
        };

        let queue = unsafe { device.get_device_queue(compute_family_index, 0) };

        Ok(Arc::new(Self {
            raw: device,
            physical_device: pdev,
            details,
            compute_queue: queue,
            compute_queue_family: compute_family_index,
            non_coherent_atom_size,
        }))
    }

    /// Create a Vulkan 1.2 timeline semaphore
    pub fn create_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<vk::Semaphore, CoreError> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);

        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        unsafe {
            self.raw.create_semaphore(&sem_info, None).map_err(|e| {
                CoreError::DriverError(format!("create_timeline_semaphore failed: {:?}", e))
            })
        }
    }

    /// Wait for a timeline semaphore to reach or exceed a target value
    pub fn wait_timeline_semaphore(
        &self,
        semaphore: vk::Semaphore,
        value: u64,
        timeout_nanos: u64,
    ) -> Result<(), CoreError> {
        let sems = [semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&values);

        unsafe {
            self.raw
                .wait_semaphores(&wait_info, timeout_nanos)
                .map_err(|e| {
                    CoreError::SyncError(format!(
                        "wait_timeline_semaphore timed out/failed: {:?}",
                        e
                    ))
                })
        }
    }

    /// Get current counter value of a timeline semaphore
    pub fn get_timeline_semaphore_value(&self, semaphore: vk::Semaphore) -> Result<u64, CoreError> {
        unsafe {
            self.raw
                .get_semaphore_counter_value(semaphore)
                .map_err(|e| {
                    CoreError::SyncError(format!("get_semaphore_counter_value failed: {:?}", e))
                })
        }
    }

    /// Synchronously copy buffer memory on this device via compute/transfer queue
    pub fn copy_buffer(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        size_bytes: u64,
    ) -> Result<f64, CoreError> {
        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let cmd_pool = unsafe {
            self.raw
                .create_command_pool(&cmd_pool_info, None)
                .map_err(|e| {
                    CoreError::DriverError(format!("create_command_pool failed: {:?}", e))
                })?
        };

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffer = unsafe {
            match self.raw.allocate_command_buffers(&cmd_alloc_info) {
                Ok(cbs) => cbs[0],
                Err(e) => {
                    self.raw.destroy_command_pool(cmd_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "allocate_command_buffers failed: {:?}",
                        e
                    )));
                }
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: size_bytes,
        };

        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::HOST_READ
                    | vk::AccessFlags::HOST_WRITE,
            );

        unsafe {
            let _ = self.raw.begin_command_buffer(cmd_buffer, &begin_info);
            self.raw
                .cmd_copy_buffer(cmd_buffer, src, dst, &[copy_region]);
            self.raw.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
            let _ = self.raw.end_command_buffer(cmd_buffer);
        }

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            match self.raw.create_fence(&fence_info, None) {
                Ok(f) => f,
                Err(e) => {
                    self.raw.destroy_command_pool(cmd_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "create_fence failed: {:?}",
                        e
                    )));
                }
            }
        };

        let submit_cmd_buffers = [cmd_buffer];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&submit_cmd_buffers)];

        let timer_start = std::time::Instant::now();
        unsafe {
            self.raw
                .queue_submit(self.compute_queue, &submit_info, fence)
                .map_err(|e| {
                    CoreError::DriverError(format!("queue_submit copy failed: {:?}", e))
                })?;

            self.raw
                .wait_for_fences(&[fence], true, 10_000_000_000)
                .map_err(|e| CoreError::SyncError(format!("wait_for_fences timed out: {:?}", e)))?;
        }
        let elapsed_micros = timer_start.elapsed().as_secs_f64() * 1_000_000.0;

        unsafe {
            self.raw.destroy_fence(fence, None);
            self.raw.destroy_command_pool(cmd_pool, None);
        }

        Ok(elapsed_micros)
    }

    /// Asynchronously copy buffer memory signaling a timeline semaphore upon completion
    pub fn copy_buffer_timeline(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        size_bytes: u64,
        timeline_sem: vk::Semaphore,
        signal_val: u64,
    ) -> Result<(), CoreError> {
        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let cmd_pool = unsafe {
            self.raw
                .create_command_pool(&cmd_pool_info, None)
                .map_err(|e| {
                    CoreError::DriverError(format!("create_command_pool failed: {:?}", e))
                })?
        };

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffer = unsafe {
            match self.raw.allocate_command_buffers(&cmd_alloc_info) {
                Ok(cbs) => cbs[0],
                Err(e) => {
                    self.raw.destroy_command_pool(cmd_pool, None);
                    return Err(CoreError::DriverError(format!(
                        "allocate_command_buffers failed: {:?}",
                        e
                    )));
                }
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: size_bytes,
        };

        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::HOST_READ
                    | vk::AccessFlags::HOST_WRITE,
            );

        unsafe {
            let _ = self.raw.begin_command_buffer(cmd_buffer, &begin_info);
            self.raw
                .cmd_copy_buffer(cmd_buffer, src, dst, &[copy_region]);
            self.raw.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
            let _ = self.raw.end_command_buffer(cmd_buffer);
        }

        let submit_cmd_buffers = [cmd_buffer];
        let sems = [timeline_sem];
        let sig_vals = [signal_val];
        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&sig_vals);

        let submit_info = [vk::SubmitInfo::default()
            .command_buffers(&submit_cmd_buffers)
            .signal_semaphores(&sems)
            .push_next(&mut timeline_info)];

        unsafe {
            self.raw
                .queue_submit(self.compute_queue, &submit_info, vk::Fence::null())
                .map_err(|e| {
                    CoreError::DriverError(format!("queue_submit timeline copy failed: {:?}", e))
                })?;
        }

        // Wait for queue idle before destroying command pool (or in production pool is kept alive)
        unsafe {
            let _ = self.raw.queue_wait_idle(self.compute_queue);
            self.raw.destroy_command_pool(cmd_pool, None);
        }

        Ok(())
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.raw.destroy_device(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::VulkanContext;
    use crate::physical_device::PhysicalDeviceEnumerator;

    #[test]
    fn test_vulkan_timeline_semaphore_lifecycle() {
        let ctx = VulkanContext::new().expect("Failed to create VulkanContext");
        let devices = PhysicalDeviceEnumerator::enumerate(&ctx.instance)
            .expect("Failed to enumerate devices");
        assert!(!devices.is_empty());

        for (pdev, details) in devices {
            println!("Testing timeline semaphore on: {}", details.name);
            let dev = VulkanDevice::new(&ctx.instance, pdev, details)
                .expect("Failed to create VulkanDevice");

            let sem = dev
                .create_timeline_semaphore(0)
                .expect("Failed to create timeline semaphore");

            let val = dev
                .get_timeline_semaphore_value(sem)
                .expect("Failed to get timeline value");
            assert_eq!(val, 0);

            // Signal from host
            let sig_info = vk::SemaphoreSignalInfo::default().semaphore(sem).value(42);
            unsafe {
                dev.raw
                    .signal_semaphore(&sig_info)
                    .expect("Failed to signal semaphore from host");
            }

            let val2 = dev
                .get_timeline_semaphore_value(sem)
                .expect("Failed to get updated timeline value");
            assert_eq!(val2, 42);

            // Wait on host
            dev.wait_timeline_semaphore(sem, 42, 1_000_000_000)
                .expect("Failed to wait on timeline semaphore");

            unsafe {
                dev.raw.destroy_semaphore(sem, None);
            }
        }
    }
}
