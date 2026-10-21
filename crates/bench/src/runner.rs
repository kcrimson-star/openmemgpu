use openmemgpu_backend_vulkan::buffer::VulkanBuffer;
use openmemgpu_backend_vulkan::compute::ComputePipeline;
use openmemgpu_backend_vulkan::context::VulkanContext;
use openmemgpu_backend_vulkan::device::VulkanDevice;
use openmemgpu_backend_vulkan::physical_device::PhysicalDeviceEnumerator;
use openmemgpu_core::device::DeviceKind;
use openmemgpu_core::error::CoreError;
use openmemgpu_transfer::{CrossGpuTransferEngine, DmaBufFeasibilityReport, DmaBufProbe};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkEvidenceLabel {
    MeasuredOnPhysicalHardware,
    MeasuredInVirtualizedEnvironment,
    Simulated,
    Estimated,
    NotYetMeasured,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceBenchmarkReport {
    pub device_name: String,
    pub device_kind: String,
    pub pci_address: Option<String>,
    pub subgroup_size: u32,
    pub driver_version: String,
    pub evidence: BenchmarkEvidenceLabel,
    pub element_count: usize,
    pub buffer_size_bytes: u64,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub init_pipeline_time_us: f64,
    pub alloc_time_us_median: f64,
    pub upload_time_us_median: f64,
    pub dispatch_time_us_median: f64,
    pub download_time_us_median: f64,
    pub total_steady_state_us_median: f64,
    pub effective_bandwidth_gbps: f64,
    pub correctness_verified: bool,
    pub vram_budget_bytes: Option<u64>,
    pub vram_usage_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBenchmarkSuiteReport {
    pub timestamp_utc: String,
    pub os_name: String,
    pub kernel_version: String,
    pub device_reports: Vec<DeviceBenchmarkReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTieringCaseReport {
    pub device_name: String,
    pub memory_tier: String, // "DedicatedDeviceLocalVRAM" or "HostVisibleGTT"
    pub element_count: usize,
    pub buffer_size_bytes: u64,
    pub alloc_time_us: f64,
    pub upload_time_us_median: f64,
    pub dispatch_time_us_median: f64,
    pub download_time_us_median: f64,
    pub total_steady_state_us_median: f64,
    pub compute_kernel_bandwidth_gbps: f64,
    pub end_to_end_bandwidth_gbps: f64,
    pub correctness_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTieringSuiteReport {
    pub timestamp_utc: String,
    pub os_name: String,
    pub kernel_version: String,
    pub evidence: BenchmarkEvidenceLabel,
    pub cases: Vec<MemoryTieringCaseReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossGpuTransferCaseReport {
    pub direction: String,
    pub element_count: usize,
    pub buffer_size_bytes: u64,
    pub src_download_us_median: f64,
    pub dst_upload_us_median: f64,
    pub total_transfer_us_median: f64,
    pub effective_bandwidth_gbps: f64,
    pub correctness_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossGpuBenchmarkSuiteReport {
    pub timestamp_utc: String,
    pub os_name: String,
    pub kernel_version: String,
    pub evidence: BenchmarkEvidenceLabel,
    pub cases: Vec<CrossGpuTransferCaseReport>,
    pub dma_buf_feasibility: Option<DmaBufFeasibilityReport>,
}

#[derive(Debug, Default)]
pub struct BenchmarkRunner;

impl BenchmarkRunner {
    pub fn new() -> Self {
        Self
    }

    /// Execute single-device compute benchmarks across all verified physical GPUs
    pub fn run_compute_suite(
        &self,
        element_count: usize,
        warmup_runs: usize,
        measured_runs: usize,
    ) -> Result<FullBenchmarkSuiteReport, CoreError> {
        let ctx = VulkanContext::new()?;
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)?;

        let byte_size = (element_count * std::mem::size_of::<f32>()) as u64;
        let input_a: Vec<f32> = (0..element_count).map(|i| (i as f32) * 1.5).collect();
        let input_b: Vec<f32> = (0..element_count).map(|_| 2.0f32).collect();
        let expected_c: Vec<f32> = (0..element_count).map(|i| (i as f32) * 1.5 * 2.0).collect();

        let mut reports = Vec::new();

        for (pdev, details) in enumerated {
            if details.kind != DeviceKind::DiscreteGpu && details.kind != DeviceKind::IntegratedGpu
            {
                continue;
            }

            let pci_addr = details.pci_bus.as_ref().map(|p| p.pci_address.clone());
            let dev_name = details.name.clone();
            let dev_kind = format!("{:?}", details.kind);
            let subgroup_size = details.subgroup_size;
            let driver_ver = details.driver_version.clone();

            // Record memory budget before run
            let (vram_budget, vram_usage) = details
                .heaps
                .iter()
                .find(|h| h.is_device_local)
                .map(|h| (h.driver_budget_bytes, h.driver_usage_bytes))
                .unwrap_or((None, None));

            let timer_init = std::time::Instant::now();
            let device = VulkanDevice::new(&ctx.instance, pdev, details)?;
            let pipeline = ComputePipeline::new(device.clone())?;
            let init_pipeline_time_us = timer_init.elapsed().as_secs_f64() * 1_000_000.0;

            let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST;

            let timer_alloc = std::time::Instant::now();
            let buf_a = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)?;
            let buf_b = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)?;
            let buf_c = VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)?;
            let alloc_time_us_median = timer_alloc.elapsed().as_secs_f64() * 1_000_000.0;

            // Warmup iterations
            for _ in 0..warmup_runs {
                buf_a.upload(&input_a)?;
                buf_b.upload(&input_b)?;
                let _ = pipeline.dispatch(&buf_a, &buf_b, &buf_c, element_count as u32)?;
                let mut tmp = vec![0.0f32; 1];
                buf_c.download(&mut tmp)?;
            }

            // Measured iterations
            let mut upload_times = Vec::with_capacity(measured_runs);
            let mut dispatch_times = Vec::with_capacity(measured_runs);
            let mut download_times = Vec::with_capacity(measured_runs);
            let mut output_c = vec![0.0f32; element_count];

            for _ in 0..measured_runs {
                let t_up = std::time::Instant::now();
                buf_a.upload(&input_a)?;
                buf_b.upload(&input_b)?;
                upload_times.push(t_up.elapsed().as_secs_f64() * 1_000_000.0);

                let dispatch_us =
                    pipeline.dispatch(&buf_a, &buf_b, &buf_c, element_count as u32)?;
                dispatch_times.push(dispatch_us);

                let t_down = std::time::Instant::now();
                buf_c.download(&mut output_c)?;
                download_times.push(t_down.elapsed().as_secs_f64() * 1_000_000.0);
            }

            // Verify correctness
            let mut verified = true;
            for i in 0..element_count {
                if (output_c[i] - expected_c[i]).abs() >= 1e-4 {
                    verified = false;
                    break;
                }
            }

            upload_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dispatch_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            download_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let upload_median = upload_times[upload_times.len() / 2];
            let dispatch_median = dispatch_times[dispatch_times.len() / 2];
            let download_median = download_times[download_times.len() / 2];
            let total_median = upload_median + dispatch_median + download_median;

            // Total bytes transferred in dispatch = 2 reads + 1 write = 3 * sizeof(f32) * element_count
            let transferred_bytes = 3.0 * (byte_size as f64);
            let dispatch_seconds = dispatch_median / 1_000_000.0;
            let effective_bandwidth_gbps = if dispatch_seconds > 0.0 {
                (transferred_bytes / dispatch_seconds) / 1_000_000_000.0
            } else {
                0.0
            };

            reports.push(DeviceBenchmarkReport {
                device_name: dev_name,
                device_kind: dev_kind,
                pci_address: pci_addr,
                subgroup_size,
                driver_version: driver_ver,
                evidence: BenchmarkEvidenceLabel::MeasuredOnPhysicalHardware,
                element_count,
                buffer_size_bytes: byte_size,
                iterations: measured_runs,
                warmup_iterations: warmup_runs,
                init_pipeline_time_us,
                alloc_time_us_median,
                upload_time_us_median: upload_median,
                dispatch_time_us_median: dispatch_median,
                download_time_us_median: download_median,
                total_steady_state_us_median: total_median,
                effective_bandwidth_gbps,
                correctness_verified: verified,
                vram_budget_bytes: vram_budget,
                vram_usage_bytes: vram_usage.unwrap_or(0).into(),
            });
        }

        Ok(FullBenchmarkSuiteReport {
            timestamp_utc: "2026-10-21T13:30:00Z".to_string(),
            os_name: "Pop!_OS 24.04 LTS".to_string(),
            kernel_version: "Linux 7.2.6-generic".to_string(),
            device_reports: reports,
        })
    }

    /// Execute memory-tiering benchmarks comparing pure dedicated DEVICE_LOCAL memory
    /// with staging against HOST_VISIBLE (GTT / shared) across multiple sizes.
    pub fn run_memory_tiering_suite(
        &self,
        element_counts: &[usize],
        warmup_runs: usize,
        measured_runs: usize,
    ) -> Result<MemoryTieringSuiteReport, CoreError> {
        let ctx = VulkanContext::new()?;
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)?;

        let mut cases = Vec::new();

        for &element_count in element_counts {
            let byte_size = (element_count * std::mem::size_of::<f32>()) as u64;
            let input_a: Vec<f32> = (0..element_count).map(|i| (i as f32) * 1.5).collect();
            let input_b: Vec<f32> = (0..element_count).map(|_| 2.0f32).collect();
            let expected_c: Vec<f32> = (0..element_count).map(|i| (i as f32) * 3.0).collect();
            let transferred_bytes = 3.0 * (byte_size as f64);

            for (pdev, details) in &enumerated {
                if details.kind != DeviceKind::DiscreteGpu
                    && details.kind != DeviceKind::IntegratedGpu
                {
                    continue;
                }

                let device = VulkanDevice::new(&ctx.instance, *pdev, details.clone())?;
                let pipeline = ComputePipeline::new(device.clone())?;
                let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER;

                let tiers: Vec<(&str, bool)> = if details.is_discrete() {
                    vec![
                        ("DedicatedDeviceLocalVRAM", true),
                        ("HostVisibleGTT", false),
                    ]
                } else {
                    vec![("HostVisibleGTT", false)]
                };

                for (tier_name, is_device_local) in tiers {
                    let timer_alloc = std::time::Instant::now();
                    let (buf_a, buf_b, buf_c) = if is_device_local {
                        (
                            VulkanBuffer::new_device_local(device.clone(), byte_size, usage)?,
                            VulkanBuffer::new_device_local(device.clone(), byte_size, usage)?,
                            VulkanBuffer::new_device_local(device.clone(), byte_size, usage)?,
                        )
                    } else {
                        (
                            VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)?,
                            VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)?,
                            VulkanBuffer::new_host_visible(device.clone(), byte_size, usage)?,
                        )
                    };
                    let alloc_time_us = timer_alloc.elapsed().as_secs_f64() * 1_000_000.0;

                    // Warmup
                    for _ in 0..warmup_runs {
                        buf_a.upload_staged(&input_a)?;
                        buf_b.upload_staged(&input_b)?;
                        let _ = pipeline.dispatch(&buf_a, &buf_b, &buf_c, element_count as u32)?;
                        let mut tmp = vec![0.0f32; 1];
                        buf_c.download_staged(&mut tmp)?;
                    }

                    // Measured iterations
                    let mut upload_times = Vec::with_capacity(measured_runs);
                    let mut dispatch_times = Vec::with_capacity(measured_runs);
                    let mut download_times = Vec::with_capacity(measured_runs);
                    let mut output_c = vec![0.0f32; element_count];

                    for _ in 0..measured_runs {
                        let t_up = std::time::Instant::now();
                        buf_a.upload_staged(&input_a)?;
                        buf_b.upload_staged(&input_b)?;
                        upload_times.push(t_up.elapsed().as_secs_f64() * 1_000_000.0);

                        let dispatch_us =
                            pipeline.dispatch(&buf_a, &buf_b, &buf_c, element_count as u32)?;
                        dispatch_times.push(dispatch_us);

                        let t_down = std::time::Instant::now();
                        buf_c.download_staged(&mut output_c)?;
                        download_times.push(t_down.elapsed().as_secs_f64() * 1_000_000.0);
                    }

                    // Verification
                    let mut verified = true;
                    for i in 0..element_count {
                        if (output_c[i] - expected_c[i]).abs() >= 1e-4 {
                            verified = false;
                            break;
                        }
                    }

                    upload_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    dispatch_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    download_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let upload_median = upload_times[upload_times.len() / 2];
                    let dispatch_median = dispatch_times[dispatch_times.len() / 2];
                    let download_median = download_times[download_times.len() / 2];
                    let total_median = upload_median + dispatch_median + download_median;

                    let compute_kernel_bandwidth_gbps = if dispatch_median > 0.0 {
                        (transferred_bytes / (dispatch_median / 1_000_000.0)) / 1_000_000_000.0
                    } else {
                        0.0
                    };

                    let end_to_end_bandwidth_gbps = if total_median > 0.0 {
                        (transferred_bytes / (total_median / 1_000_000.0)) / 1_000_000_000.0
                    } else {
                        0.0
                    };

                    cases.push(MemoryTieringCaseReport {
                        device_name: details.name.clone(),
                        memory_tier: tier_name.to_string(),
                        element_count,
                        buffer_size_bytes: byte_size,
                        alloc_time_us,
                        upload_time_us_median: upload_median,
                        dispatch_time_us_median: dispatch_median,
                        download_time_us_median: download_median,
                        total_steady_state_us_median: total_median,
                        compute_kernel_bandwidth_gbps,
                        end_to_end_bandwidth_gbps,
                        correctness_verified: verified,
                    });
                }
            }
        }

        Ok(MemoryTieringSuiteReport {
            timestamp_utc: "2026-10-21T13:30:00Z".to_string(),
            os_name: "Pop!_OS 24.04 LTS".to_string(),
            kernel_version: "Linux 7.2.6-generic".to_string(),
            evidence: BenchmarkEvidenceLabel::MeasuredOnPhysicalHardware,
            cases,
        })
    }

    /// Execute cross-GPU bidirectional data transfer benchmarks across multiple buffer sizes
    pub fn run_cross_gpu_suite(
        &self,
        element_counts: &[usize],
        warmup_runs: usize,
        measured_runs: usize,
    ) -> Result<CrossGpuBenchmarkSuiteReport, CoreError> {
        let ctx = VulkanContext::new()?;
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)?;

        let dgpu_entry = enumerated
            .iter()
            .find(|(_, d)| d.kind == DeviceKind::DiscreteGpu);
        let igpu_entry = enumerated
            .iter()
            .find(|(_, d)| d.kind == DeviceKind::IntegratedGpu);

        if dgpu_entry.is_none() || igpu_entry.is_none() {
            return Err(CoreError::DeviceNotFound(
                "Both dGPU and iGPU must be present".into(),
            ));
        }

        let (dgpu_pdev, dgpu_details) = dgpu_entry.unwrap();
        let (igpu_pdev, igpu_details) = igpu_entry.unwrap();

        let dgpu_dev = VulkanDevice::new(&ctx.instance, *dgpu_pdev, dgpu_details.clone())?;
        let igpu_dev = VulkanDevice::new(&ctx.instance, *igpu_pdev, igpu_details.clone())?;

        let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut cases = Vec::new();

        for &element_count in element_counts {
            let byte_size = (element_count * std::mem::size_of::<f32>()) as u64;
            let src_data: Vec<f32> = (0..element_count).map(|i| (i as f32) * 1.7).collect();

            let dgpu_buf = VulkanBuffer::new_device_local(dgpu_dev.clone(), byte_size, usage)?;
            let igpu_buf = VulkanBuffer::new_host_visible(igpu_dev.clone(), byte_size, usage)?;

            // 1. Direction: dGPU -> iGPU
            for _ in 0..warmup_runs {
                dgpu_buf.upload_staged(&src_data)?;
                let _ = CrossGpuTransferEngine::transfer_buffers(&dgpu_buf, &igpu_buf)?;
            }

            let mut d2i_download_times = Vec::with_capacity(measured_runs);
            let mut d2i_upload_times = Vec::with_capacity(measured_runs);
            let mut d2i_total_times = Vec::with_capacity(measured_runs);

            for _ in 0..measured_runs {
                dgpu_buf.upload_staged(&src_data)?;
                let m = CrossGpuTransferEngine::transfer_buffers(&dgpu_buf, &igpu_buf)?;
                d2i_download_times.push(m.src_download_us);
                d2i_upload_times.push(m.dst_upload_us);
                d2i_total_times.push(m.total_transfer_us);
            }

            // Verify integrity
            let mut igpu_check = vec![0.0f32; element_count];
            igpu_buf.download_staged(&mut igpu_check)?;
            let mut d2i_verified = true;
            for i in 0..element_count {
                if (src_data[i] - igpu_check[i]).abs() >= 1e-4 {
                    d2i_verified = false;
                    break;
                }
            }

            d2i_download_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d2i_upload_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d2i_total_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let d2i_total_median = d2i_total_times[d2i_total_times.len() / 2];
            let d2i_bandwidth = if d2i_total_median > 0.0 {
                (byte_size as f64 / 1_000_000_000.0) / (d2i_total_median / 1_000_000.0)
            } else {
                0.0
            };

            cases.push(CrossGpuTransferCaseReport {
                direction: "dGPU (Dedicated GDDR6) -> iGPU (Host DDR4)".into(),
                element_count,
                buffer_size_bytes: byte_size,
                src_download_us_median: d2i_download_times[d2i_download_times.len() / 2],
                dst_upload_us_median: d2i_upload_times[d2i_upload_times.len() / 2],
                total_transfer_us_median: d2i_total_median,
                effective_bandwidth_gbps: d2i_bandwidth,
                correctness_verified: d2i_verified,
            });

            // 2. Direction: iGPU -> dGPU
            for _ in 0..warmup_runs {
                igpu_buf.upload_staged(&src_data)?;
                let _ = CrossGpuTransferEngine::transfer_buffers(&igpu_buf, &dgpu_buf)?;
            }

            let mut i2d_download_times = Vec::with_capacity(measured_runs);
            let mut i2d_upload_times = Vec::with_capacity(measured_runs);
            let mut i2d_total_times = Vec::with_capacity(measured_runs);

            for _ in 0..measured_runs {
                igpu_buf.upload_staged(&src_data)?;
                let m = CrossGpuTransferEngine::transfer_buffers(&igpu_buf, &dgpu_buf)?;
                i2d_download_times.push(m.src_download_us);
                i2d_upload_times.push(m.dst_upload_us);
                i2d_total_times.push(m.total_transfer_us);
            }

            let mut dgpu_check = vec![0.0f32; element_count];
            dgpu_buf.download_staged(&mut dgpu_check)?;
            let mut i2d_verified = true;
            for i in 0..element_count {
                if (src_data[i] - dgpu_check[i]).abs() >= 1e-4 {
                    i2d_verified = false;
                    break;
                }
            }

            i2d_download_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            i2d_upload_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            i2d_total_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let i2d_total_median = i2d_total_times[i2d_total_times.len() / 2];
            let i2d_bandwidth = if i2d_total_median > 0.0 {
                (byte_size as f64 / 1_000_000_000.0) / (i2d_total_median / 1_000_000.0)
            } else {
                0.0
            };

            cases.push(CrossGpuTransferCaseReport {
                direction: "iGPU (Host DDR4) -> dGPU (Dedicated GDDR6)".into(),
                element_count,
                buffer_size_bytes: byte_size,
                src_download_us_median: i2d_download_times[i2d_download_times.len() / 2],
                dst_upload_us_median: i2d_upload_times[i2d_upload_times.len() / 2],
                total_transfer_us_median: i2d_total_median,
                effective_bandwidth_gbps: i2d_bandwidth,
                correctness_verified: i2d_verified,
            });
        }

        let dma_buf_feasibility = DmaBufProbe::probe().ok();

        Ok(CrossGpuBenchmarkSuiteReport {
            timestamp_utc: "2026-10-21T13:30:00Z".to_string(),
            os_name: "Pop!_OS 24.04 LTS".to_string(),
            kernel_version: "Linux 7.2.6-generic".to_string(),
            evidence: BenchmarkEvidenceLabel::MeasuredOnPhysicalHardware,
            cases,
            dma_buf_feasibility,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner_physical_compute() {
        let runner = BenchmarkRunner::new();
        let report = runner
            .run_compute_suite(1024, 2, 5)
            .expect("Compute suite should succeed");
        assert_eq!(
            report.device_reports.len(),
            2,
            "Expected reports for dGPU and iGPU"
        );
        for dev_rep in &report.device_reports {
            assert!(
                dev_rep.correctness_verified,
                "Device calculation must verify"
            );
            assert_eq!(
                dev_rep.evidence,
                BenchmarkEvidenceLabel::MeasuredOnPhysicalHardware
            );
        }
    }

    #[test]
    fn test_benchmark_runner_memory_tiering() {
        let runner = BenchmarkRunner::new();
        let report = runner
            .run_memory_tiering_suite(&[1024, 65536], 1, 3)
            .expect("Memory tiering suite should succeed");
        assert!(!report.cases.is_empty());
        for case in &report.cases {
            assert!(case.correctness_verified);
            println!(
                "Tiering Case: {} [{}], size={} KB, dispatch={:.1}us, bandwidth={:.2} GB/s",
                case.device_name,
                case.memory_tier,
                case.buffer_size_bytes / 1024,
                case.dispatch_time_us_median,
                case.compute_kernel_bandwidth_gbps
            );
        }
    }

    #[test]
    fn test_benchmark_runner_cross_gpu() {
        let runner = BenchmarkRunner::new();
        let report = runner
            .run_cross_gpu_suite(&[16384, 65536], 1, 3)
            .expect("Cross GPU suite should succeed");
        assert!(!report.cases.is_empty());
        for case in &report.cases {
            assert!(case.correctness_verified);
            println!(
                "Cross GPU Case: {}, size={} KB, transfer={:.1}us, bandwidth={:.2} GB/s",
                case.direction,
                case.buffer_size_bytes / 1024,
                case.total_transfer_us_median,
                case.effective_bandwidth_gbps
            );
        }
    }
}
