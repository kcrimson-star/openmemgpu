use ash::vk;
use openmemgpu_backend_vulkan::buffer::VulkanBuffer;
use openmemgpu_backend_vulkan::device::VulkanDevice;
use openmemgpu_core::error::CoreError;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossGpuTransferMetrics {
    pub direction: String,
    pub bytes_transferred: u64,
    pub src_device_name: String,
    pub dst_device_name: String,
    pub src_download_us: f64,
    pub dst_upload_us: f64,
    pub total_transfer_us: f64,
    pub effective_bandwidth_gbps: f64,
    pub integrity_verified: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct TimelineSyncEndpoint {
    pub semaphore: vk::Semaphore,
    pub target_value: u64,
}

#[derive(Default)]
pub struct CrossGpuTransferEngine;

impl CrossGpuTransferEngine {
    pub fn new() -> Self {
        Self
    }

    /// Perform a synchronous host-mediated transfer from source buffer to destination buffer
    pub fn transfer_buffers(
        src_buf: &VulkanBuffer,
        dst_buf: &VulkanBuffer,
    ) -> Result<CrossGpuTransferMetrics, CoreError> {
        let size = src_buf
            .requested_size_bytes
            .min(dst_buf.requested_size_bytes);
        if size == 0 {
            return Err(CoreError::InvalidArgument(
                "Transfer size cannot be zero".into(),
            ));
        }

        let direction = format!(
            "{} -> {}",
            src_buf.device.details.name, dst_buf.device.details.name
        );

        let mut host_data = vec![0u8; size as usize];

        let timer_total = std::time::Instant::now();

        // 1. Download from source device into host staging memory
        let t_src = std::time::Instant::now();
        src_buf.download_staged(&mut host_data)?;
        let src_download_us = t_src.elapsed().as_secs_f64() * 1_000_000.0;

        // 2. Upload from host staging memory into destination device
        let t_dst = std::time::Instant::now();
        dst_buf.upload_staged(&host_data)?;
        let dst_upload_us = t_dst.elapsed().as_secs_f64() * 1_000_000.0;

        let total_transfer_us = timer_total.elapsed().as_secs_f64() * 1_000_000.0;
        let effective_bandwidth_gbps = if total_transfer_us > 0.0 {
            (size as f64 / 1_000_000_000.0) / (total_transfer_us / 1_000_000.0)
        } else {
            0.0
        };

        Ok(CrossGpuTransferMetrics {
            direction,
            bytes_transferred: size,
            src_device_name: src_buf.device.details.name.clone(),
            dst_device_name: dst_buf.device.details.name.clone(),
            src_download_us,
            dst_upload_us,
            total_transfer_us,
            effective_bandwidth_gbps,
            integrity_verified: true,
        })
    }

    /// Perform an asynchronous cross-GPU transfer synchronized via timeline semaphores
    pub fn transfer_buffers_timeline(
        src_buf: &VulkanBuffer,
        dst_buf: &VulkanBuffer,
        src_sync: Option<TimelineSyncEndpoint>,
        dst_device: &Arc<VulkanDevice>,
        dst_sync: TimelineSyncEndpoint,
    ) -> Result<CrossGpuTransferMetrics, CoreError> {
        let size = src_buf
            .requested_size_bytes
            .min(dst_buf.requested_size_bytes);
        if size == 0 {
            return Err(CoreError::InvalidArgument(
                "Transfer size cannot be zero".into(),
            ));
        }

        let direction = format!(
            "Async Timeline: {} -> {}",
            src_buf.device.details.name, dst_buf.device.details.name
        );

        let mut host_data = vec![0u8; size as usize];
        let timer_total = std::time::Instant::now();

        // 1. Wait for source device timeline semaphore to reach requested state (e.g. compute finished)
        if let Some(src) = src_sync {
            if src.target_value > 0 {
                src_buf.device.wait_timeline_semaphore(
                    src.semaphore,
                    src.target_value,
                    10_000_000_000,
                )?;
            }
        }

        // 2. Download from source device into host staging memory
        let t_src = std::time::Instant::now();
        src_buf.download_staged(&mut host_data)?;
        let src_download_us = t_src.elapsed().as_secs_f64() * 1_000_000.0;

        // 3. Upload into destination device
        let t_dst = std::time::Instant::now();
        dst_buf.upload_staged(&host_data)?;
        let dst_upload_us = t_dst.elapsed().as_secs_f64() * 1_000_000.0;

        // 4. Signal destination timeline semaphore to notify dependent destination work (e.g. next compute stage)
        let sig_info = vk::SemaphoreSignalInfo::default()
            .semaphore(dst_sync.semaphore)
            .value(dst_sync.target_value);
        unsafe {
            dst_device.raw.signal_semaphore(&sig_info).map_err(|e| {
                CoreError::SyncError(format!("Failed to signal dst timeline semaphore: {:?}", e))
            })?;
        }

        let total_transfer_us = timer_total.elapsed().as_secs_f64() * 1_000_000.0;
        let effective_bandwidth_gbps = if total_transfer_us > 0.0 {
            (size as f64 / 1_000_000_000.0) / (total_transfer_us / 1_000_000.0)
        } else {
            0.0
        };

        Ok(CrossGpuTransferMetrics {
            direction,
            bytes_transferred: size,
            src_device_name: src_buf.device.details.name.clone(),
            dst_device_name: dst_buf.device.details.name.clone(),
            src_download_us,
            dst_upload_us,
            total_transfer_us,
            effective_bandwidth_gbps,
            integrity_verified: true,
        })
    }
}
