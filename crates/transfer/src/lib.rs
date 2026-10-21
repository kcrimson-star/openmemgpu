pub mod cross_gpu;
pub mod dma_buf;
pub mod staging;

pub use cross_gpu::{CrossGpuTransferEngine, CrossGpuTransferMetrics, TimelineSyncEndpoint};
pub use dma_buf::{DeviceDmaBufCapabilities, DmaBufFeasibilityReport, DmaBufProbe};
pub use staging::StagingBufferManager;

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;
    use openmemgpu_backend_vulkan::buffer::VulkanBuffer;
    use openmemgpu_backend_vulkan::context::VulkanContext;
    use openmemgpu_backend_vulkan::device::VulkanDevice;
    use openmemgpu_backend_vulkan::physical_device::PhysicalDeviceEnumerator;
    use openmemgpu_core::device::DeviceKind;

    #[test]
    fn test_dma_buf_feasibility_probe() {
        let report = DmaBufProbe::probe().expect("DmaBufProbe::probe failed");
        println!("DMA-BUF Feasibility Probe Report:\n{:#?}", report);
        assert!(!report.devices.is_empty());
    }

    #[test]
    fn test_bidirectional_cross_gpu_transfer_hardware() {
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
            println!("Skipping cross-GPU test: require both dGPU and iGPU");
            return;
        }

        let (dgpu_pdev, dgpu_details) = dgpu_entry.unwrap();
        let (igpu_pdev, igpu_details) = igpu_entry.unwrap();

        let dgpu_device = VulkanDevice::new(&ctx.instance, *dgpu_pdev, dgpu_details.clone())
            .expect("Failed to create dGPU VulkanDevice");
        let igpu_device = VulkanDevice::new(&ctx.instance, *igpu_pdev, igpu_details.clone())
            .expect("Failed to create iGPU VulkanDevice");

        let element_count = 65536usize; // 256 KB
        let byte_size = (element_count * std::mem::size_of::<f32>()) as u64;

        let original_data: Vec<f32> = (0..element_count)
            .map(|i| (i as f32) * std::f32::consts::PI)
            .collect();

        // 1. Allocate dGPU buffer (Device Local dedicated GDDR6)
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let dgpu_buf = VulkanBuffer::new_device_local(dgpu_device.clone(), byte_size, usage)
            .expect("Failed to allocate dGPU device-local buffer");
        assert!(dgpu_buf.is_device_local);

        // Upload initial data to dGPU
        dgpu_buf
            .upload_staged(&original_data)
            .expect("Upload to dGPU failed");

        // 2. Allocate iGPU buffer (Host-visible / APU memory)
        let igpu_buf = VulkanBuffer::new_host_visible(igpu_device.clone(), byte_size, usage)
            .expect("Failed to allocate iGPU host-visible buffer");

        // 3. Transfer Direction 1: dGPU -> iGPU
        let metrics_d2i = CrossGpuTransferEngine::transfer_buffers(&dgpu_buf, &igpu_buf)
            .expect("Transfer dGPU -> iGPU failed");
        println!("Direction 1 (dGPU -> iGPU): {:#?}", metrics_d2i);

        // Verify data integrity on iGPU
        let mut igpu_readback = vec![0.0f32; element_count];
        igpu_buf
            .download_staged(&mut igpu_readback)
            .expect("Download from iGPU failed");
        for i in 0..element_count {
            assert_eq!(
                original_data[i], igpu_readback[i],
                "Mismatch at index {}",
                i
            );
        }

        // 4. Modify data on iGPU and Transfer Direction 2: iGPU -> dGPU
        let modified_data: Vec<f32> = (0..element_count).map(|i| (i as f32) * 2.71).collect();
        igpu_buf
            .upload_staged(&modified_data)
            .expect("Upload to iGPU failed");

        let metrics_i2d = CrossGpuTransferEngine::transfer_buffers(&igpu_buf, &dgpu_buf)
            .expect("Transfer iGPU -> dGPU failed");
        println!("Direction 2 (iGPU -> dGPU): {:#?}", metrics_i2d);

        // Verify data integrity on dGPU
        let mut dgpu_readback = vec![0.0f32; element_count];
        dgpu_buf
            .download_staged(&mut dgpu_readback)
            .expect("Download from dGPU failed");
        for i in 0..element_count {
            assert_eq!(
                modified_data[i], dgpu_readback[i],
                "Mismatch at index {}",
                i
            );
        }

        // 5. Asynchronous transfer with timeline semaphores
        let dgpu_sem = dgpu_device.create_timeline_semaphore(0).unwrap();
        let igpu_sem = igpu_device.create_timeline_semaphore(0).unwrap();

        // Host signals dgpu_sem = 1 (representing upstream compute finished)
        let sig_info = vk::SemaphoreSignalInfo::default()
            .semaphore(dgpu_sem)
            .value(1);
        unsafe {
            dgpu_device.raw.signal_semaphore(&sig_info).unwrap();
        }

        let async_metrics = CrossGpuTransferEngine::transfer_buffers_timeline(
            &dgpu_buf,
            &igpu_buf,
            Some(TimelineSyncEndpoint {
                semaphore: dgpu_sem,
                target_value: 1,
            }),
            &igpu_device,
            TimelineSyncEndpoint {
                semaphore: igpu_sem,
                target_value: 100,
            },
        )
        .expect("Async timeline transfer failed");
        println!("Async Timeline Transfer: {:#?}", async_metrics);

        // Verify destination semaphore reached 100
        let igpu_sem_val = igpu_device.get_timeline_semaphore_value(igpu_sem).unwrap();
        assert_eq!(igpu_sem_val, 100);

        unsafe {
            dgpu_device.raw.destroy_semaphore(dgpu_sem, None);
            igpu_device.raw.destroy_semaphore(igpu_sem, None);
        }

        println!(
            "Cross-GPU transfers and timeline synchronization fully validated on real hardware!"
        );
    }
}
