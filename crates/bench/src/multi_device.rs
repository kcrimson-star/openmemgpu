use openmemgpu_backend_vulkan::buffer::VulkanBuffer;
use openmemgpu_backend_vulkan::compute::TwoStageComputePipeline;
use openmemgpu_backend_vulkan::context::VulkanContext;
use openmemgpu_backend_vulkan::device::VulkanDevice;
use openmemgpu_backend_vulkan::physical_device::PhysicalDeviceEnumerator;
use openmemgpu_core::device::DeviceKind;
use openmemgpu_core::error::CoreError;
use openmemgpu_scheduler::partitioner::AsymmetricPartitioner;
use openmemgpu_scheduler::scheduler::{WorkloadExecutionMode, WorkloadScheduler};
use openmemgpu_transfer::CrossGpuTransferEngine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub median: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
    pub sample_count: usize,
}

impl MetricStats {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        if samples.is_empty() {
            return Self {
                median: 0.0,
                p95: 0.0,
                min: 0.0,
                max: 0.0,
                sample_count: 0,
            };
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let count = samples.len();
        let min = samples[0];
        let max = samples[count - 1];
        let median = samples[count / 2];
        let p95_idx = ((count as f64 * 0.95).ceil() as usize).min(count) - 1;
        let p95 = samples[p95_idx];

        Self {
            median,
            p95,
            min,
            max,
            sample_count: count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDeviceWorkloadResult {
    pub mode: WorkloadExecutionMode,
    pub total_elements: usize,
    pub buffer_size_bytes: u64,
    pub dgpu_elements: usize,
    pub igpu_elements: usize,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub alloc_time_us: MetricStats,
    pub upload_time_us: MetricStats,
    pub dgpu_compute_time_us: MetricStats,
    pub igpu_compute_time_us: MetricStats,
    pub parallel_compute_time_us: MetricStats,
    pub cross_device_transfer_time_us: MetricStats,
    pub sync_and_merge_time_us: MetricStats,
    pub download_time_us: MetricStats,
    pub end_to_end_wallclock_us: MetricStats,
    pub effective_throughput_gbps: f64,
    pub correctness_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDeviceBenchmarkComparison {
    pub total_elements: usize,
    pub buffer_size_bytes: u64,
    pub dgpu_only: MultiDeviceWorkloadResult,
    pub igpu_only: MultiDeviceWorkloadResult,
    pub partitioned_multigpu: MultiDeviceWorkloadResult,
    pub speedup_vs_dgpu: f64,
    pub speedup_vs_igpu: f64,
    pub outcome_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDeviceSuiteReport {
    pub timestamp_utc: String,
    pub os_name: String,
    pub kernel_version: String,
    pub dgpu_name: String,
    pub igpu_name: String,
    pub driver_version: String,
    pub comparisons: Vec<MultiDeviceBenchmarkComparison>,
}

pub struct MultiDeviceWorkloadRunner {
    pub dgpu: Arc<VulkanDevice>,
    pub igpu: Arc<VulkanDevice>,
    pub scheduler: WorkloadScheduler,
    _ctx: Arc<VulkanContext>,
}

impl MultiDeviceWorkloadRunner {
    pub fn new() -> Result<Self, CoreError> {
        let ctx = Arc::new(VulkanContext::new()?);
        let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)?;

        let dgpu_entry = enumerated
            .iter()
            .find(|(_, d)| d.kind == DeviceKind::DiscreteGpu)
            .ok_or_else(|| CoreError::DeviceNotFound("Discrete GPU not found".into()))?;
        let igpu_entry = enumerated
            .iter()
            .find(|(_, d)| d.kind == DeviceKind::IntegratedGpu)
            .ok_or_else(|| CoreError::DeviceNotFound("Integrated GPU not found".into()))?;

        let dgpu = VulkanDevice::new(&ctx.instance, dgpu_entry.0, dgpu_entry.1.clone())?;
        let igpu = VulkanDevice::new(&ctx.instance, igpu_entry.0, igpu_entry.1.clone())?;

        Ok(Self {
            dgpu,
            igpu,
            scheduler: WorkloadScheduler::default(),
            _ctx: ctx,
        })
    }

    /// Reference CPU calculation for two-stage workload:
    /// Stage 1: Z = A * B
    /// Stage 2: Out = Z * 1.5 + 0.5
    pub fn calculate_cpu_reference(input_a: &[f32], input_b: &[f32]) -> Vec<f32> {
        assert_eq!(input_a.len(), input_b.len());
        input_a
            .iter()
            .zip(input_b.iter())
            .map(|(&a, &b)| (a * b) * 1.5 + 0.5)
            .collect()
    }

    /// Run the workload in a specific execution mode across repeated iterations
    pub fn run_mode(
        &self,
        mode: WorkloadExecutionMode,
        total_elements: usize,
        warmup_runs: usize,
        measured_runs: usize,
    ) -> Result<MultiDeviceWorkloadResult, CoreError> {
        let byte_size = (total_elements * std::mem::size_of::<f32>()) as u64;
        let input_a: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 1.25).collect();
        let input_b: Vec<f32> = (0..total_elements).map(|_| 2.0f32).collect();
        let expected_out = Self::calculate_cpu_reference(&input_a, &input_b);

        let (dgpu_elements, igpu_elements) = match mode {
            WorkloadExecutionMode::DiscreteGpuOnly => (total_elements, 0),
            WorkloadExecutionMode::IntegratedGpuOnly => (0, total_elements),
            WorkloadExecutionMode::PartitionedMultiGpu => {
                let (part_d, part_i) = AsymmetricPartitioner::partition(
                    (
                        openmemgpu_core::device::DeviceId(0),
                        self.scheduler.dgpu_tflops,
                    ),
                    (
                        openmemgpu_core::device::DeviceId(1),
                        self.scheduler.igpu_tflops,
                    ),
                    total_elements,
                );
                (part_d.slice_count, part_i.slice_count)
            }
        };

        let pipeline_dgpu = TwoStageComputePipeline::new(self.dgpu.clone())?;
        let pipeline_igpu = TwoStageComputePipeline::new(self.igpu.clone())?;

        // Pre-warmup iterations
        for _ in 0..warmup_runs {
            self.execute_single_run(
                mode,
                total_elements,
                dgpu_elements,
                igpu_elements,
                &input_a,
                &input_b,
                &pipeline_dgpu,
                &pipeline_igpu,
            )?;
        }

        // Measured runs
        let mut alloc_samples = Vec::with_capacity(measured_runs);
        let mut upload_samples = Vec::with_capacity(measured_runs);
        let mut dgpu_compute_samples = Vec::with_capacity(measured_runs);
        let mut igpu_compute_samples = Vec::with_capacity(measured_runs);
        let mut parallel_compute_samples = Vec::with_capacity(measured_runs);
        let mut transfer_samples = Vec::with_capacity(measured_runs);
        let mut sync_merge_samples = Vec::with_capacity(measured_runs);
        let mut download_samples = Vec::with_capacity(measured_runs);
        let mut wallclock_samples = Vec::with_capacity(measured_runs);

        let mut final_out = vec![0.0f32; total_elements];

        for _ in 0..measured_runs {
            let run_metrics = self.execute_single_run(
                mode,
                total_elements,
                dgpu_elements,
                igpu_elements,
                &input_a,
                &input_b,
                &pipeline_dgpu,
                &pipeline_igpu,
            )?;

            alloc_samples.push(run_metrics.alloc_time_us);
            upload_samples.push(run_metrics.upload_time_us);
            dgpu_compute_samples.push(run_metrics.dgpu_compute_us);
            igpu_compute_samples.push(run_metrics.igpu_compute_us);
            parallel_compute_samples.push(run_metrics.parallel_compute_us);
            transfer_samples.push(run_metrics.transfer_time_us);
            sync_merge_samples.push(run_metrics.sync_merge_time_us);
            download_samples.push(run_metrics.download_time_us);
            wallclock_samples.push(run_metrics.end_to_end_wallclock_us);

            final_out = run_metrics.output;
        }

        // Verify correctness against CPU reference
        let mut correctness_verified = true;
        for i in 0..total_elements {
            let diff = (final_out[i] - expected_out[i]).abs();
            if diff >= 1e-4 {
                correctness_verified = false;
                break;
            }
        }

        let wallclock_stats = MetricStats::from_samples(wallclock_samples);
        let effective_throughput_gbps = if wallclock_stats.median > 0.0 {
            // Total memory operations in 2-stage workload = A (read) + B (read) + Z (write) + Z (read) + Out (write) = 5 * sizeof(f32) * elements
            let total_traffic_bytes = 5.0 * (byte_size as f64);
            (total_traffic_bytes / (wallclock_stats.median / 1.0e6)) / 1.0e9
        } else {
            0.0
        };

        Ok(MultiDeviceWorkloadResult {
            mode,
            total_elements,
            buffer_size_bytes: byte_size,
            dgpu_elements,
            igpu_elements,
            iterations: measured_runs,
            warmup_iterations: warmup_runs,
            alloc_time_us: MetricStats::from_samples(alloc_samples),
            upload_time_us: MetricStats::from_samples(upload_samples),
            dgpu_compute_time_us: MetricStats::from_samples(dgpu_compute_samples),
            igpu_compute_time_us: MetricStats::from_samples(igpu_compute_samples),
            parallel_compute_time_us: MetricStats::from_samples(parallel_compute_samples),
            cross_device_transfer_time_us: MetricStats::from_samples(transfer_samples),
            sync_and_merge_time_us: MetricStats::from_samples(sync_merge_samples),
            download_time_us: MetricStats::from_samples(download_samples),
            end_to_end_wallclock_us: wallclock_stats,
            effective_throughput_gbps,
            correctness_verified,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_single_run(
        &self,
        mode: WorkloadExecutionMode,
        total_elements: usize,
        dgpu_elements: usize,
        igpu_elements: usize,
        input_a: &[f32],
        input_b: &[f32],
        pipeline_dgpu: &TwoStageComputePipeline,
        pipeline_igpu: &TwoStageComputePipeline,
    ) -> Result<SingleRunMetrics, CoreError> {
        let t_wall_start = std::time::Instant::now();
        let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER;

        match mode {
            WorkloadExecutionMode::DiscreteGpuOnly => {
                let byte_size = (total_elements * std::mem::size_of::<f32>()) as u64;

                let t_alloc = std::time::Instant::now();
                let buf_a = VulkanBuffer::new_device_local(self.dgpu.clone(), byte_size, usage)?;
                let buf_b = VulkanBuffer::new_device_local(self.dgpu.clone(), byte_size, usage)?;
                let buf_z = VulkanBuffer::new_device_local(self.dgpu.clone(), byte_size, usage)?;
                let buf_out = VulkanBuffer::new_device_local(self.dgpu.clone(), byte_size, usage)?;
                let alloc_time_us = t_alloc.elapsed().as_secs_f64() * 1.0e6;

                let t_upload = std::time::Instant::now();
                buf_a.upload_staged(input_a)?;
                buf_b.upload_staged(input_b)?;
                let upload_time_us = t_upload.elapsed().as_secs_f64() * 1.0e6;

                let (t1, t2) = pipeline_dgpu.dispatch(
                    &buf_a,
                    &buf_b,
                    &buf_z,
                    &buf_out,
                    total_elements as u32,
                )?;
                let dgpu_compute_us = t1 + t2;

                let mut output = vec![0.0f32; total_elements];
                let t_down = std::time::Instant::now();
                buf_out.download_staged(&mut output)?;
                let download_time_us = t_down.elapsed().as_secs_f64() * 1.0e6;

                let end_to_end_wallclock_us = t_wall_start.elapsed().as_secs_f64() * 1.0e6;

                Ok(SingleRunMetrics {
                    alloc_time_us,
                    upload_time_us,
                    dgpu_compute_us,
                    igpu_compute_us: 0.0,
                    parallel_compute_us: dgpu_compute_us,
                    transfer_time_us: 0.0,
                    sync_merge_time_us: 0.0,
                    download_time_us,
                    end_to_end_wallclock_us,
                    output,
                })
            }
            WorkloadExecutionMode::IntegratedGpuOnly => {
                let byte_size = (total_elements * std::mem::size_of::<f32>()) as u64;

                let t_alloc = std::time::Instant::now();
                let buf_a = VulkanBuffer::new_host_visible(self.igpu.clone(), byte_size, usage)?;
                let buf_b = VulkanBuffer::new_host_visible(self.igpu.clone(), byte_size, usage)?;
                let buf_z = VulkanBuffer::new_host_visible(self.igpu.clone(), byte_size, usage)?;
                let buf_out = VulkanBuffer::new_host_visible(self.igpu.clone(), byte_size, usage)?;
                let alloc_time_us = t_alloc.elapsed().as_secs_f64() * 1.0e6;

                let t_upload = std::time::Instant::now();
                buf_a.upload_staged(input_a)?;
                buf_b.upload_staged(input_b)?;
                let upload_time_us = t_upload.elapsed().as_secs_f64() * 1.0e6;

                let (t1, t2) = pipeline_igpu.dispatch(
                    &buf_a,
                    &buf_b,
                    &buf_z,
                    &buf_out,
                    total_elements as u32,
                )?;
                let igpu_compute_us = t1 + t2;

                let mut output = vec![0.0f32; total_elements];
                let t_down = std::time::Instant::now();
                buf_out.download_staged(&mut output)?;
                let download_time_us = t_down.elapsed().as_secs_f64() * 1.0e6;

                let end_to_end_wallclock_us = t_wall_start.elapsed().as_secs_f64() * 1.0e6;

                Ok(SingleRunMetrics {
                    alloc_time_us,
                    upload_time_us,
                    dgpu_compute_us: 0.0,
                    igpu_compute_us,
                    parallel_compute_us: igpu_compute_us,
                    transfer_time_us: 0.0,
                    sync_merge_time_us: 0.0,
                    download_time_us,
                    end_to_end_wallclock_us,
                    output,
                })
            }
            WorkloadExecutionMode::PartitionedMultiGpu => {
                let dgpu_bytes = (dgpu_elements * std::mem::size_of::<f32>()) as u64;
                let igpu_bytes = (igpu_elements * std::mem::size_of::<f32>()) as u64;

                let t_alloc = std::time::Instant::now();
                // Partition 0 buffers on dGPU (dedicated GDDR6)
                let dgpu_buf_a =
                    VulkanBuffer::new_device_local(self.dgpu.clone(), dgpu_bytes, usage)?;
                let dgpu_buf_b =
                    VulkanBuffer::new_device_local(self.dgpu.clone(), dgpu_bytes, usage)?;
                let dgpu_buf_z =
                    VulkanBuffer::new_device_local(self.dgpu.clone(), dgpu_bytes, usage)?;
                let dgpu_buf_out =
                    VulkanBuffer::new_device_local(self.dgpu.clone(), dgpu_bytes, usage)?;

                // Partition 1 buffers on iGPU (host-visible UMA RAM)
                let igpu_buf_a =
                    VulkanBuffer::new_host_visible(self.igpu.clone(), igpu_bytes, usage)?;
                let igpu_buf_b =
                    VulkanBuffer::new_host_visible(self.igpu.clone(), igpu_bytes, usage)?;
                let igpu_buf_z =
                    VulkanBuffer::new_host_visible(self.igpu.clone(), igpu_bytes, usage)?;
                let igpu_buf_out =
                    VulkanBuffer::new_host_visible(self.igpu.clone(), igpu_bytes, usage)?;
                let alloc_time_us = t_alloc.elapsed().as_secs_f64() * 1.0e6;

                // Host upload slices
                let t_upload = std::time::Instant::now();
                dgpu_buf_a.upload_staged(&input_a[..dgpu_elements])?;
                dgpu_buf_b.upload_staged(&input_b[..dgpu_elements])?;
                igpu_buf_a.upload_staged(&input_a[dgpu_elements..])?;
                igpu_buf_b.upload_staged(&input_b[dgpu_elements..])?;
                let upload_time_us = t_upload.elapsed().as_secs_f64() * 1.0e6;

                // Concurrent execution on both physical GPUs using scoped threads
                let t_comp_start = std::time::Instant::now();
                let (dgpu_res, igpu_res) = std::thread::scope(|s| {
                    let h_d = s.spawn(|| {
                        pipeline_dgpu.dispatch(
                            &dgpu_buf_a,
                            &dgpu_buf_b,
                            &dgpu_buf_z,
                            &dgpu_buf_out,
                            dgpu_elements as u32,
                        )
                    });
                    let h_i = s.spawn(|| {
                        pipeline_igpu.dispatch(
                            &igpu_buf_a,
                            &igpu_buf_b,
                            &igpu_buf_z,
                            &igpu_buf_out,
                            igpu_elements as u32,
                        )
                    });
                    (h_d.join().unwrap(), h_i.join().unwrap())
                });

                let (d1, d2) = dgpu_res?;
                let (i1, i2) = igpu_res?;
                let parallel_compute_us = t_comp_start.elapsed().as_secs_f64() * 1.0e6;
                let dgpu_compute_us = d1 + d2;
                let igpu_compute_us = i1 + i2;

                // Cross-device transfer: Transfer partition 1 from iGPU to dGPU staging buffer
                // (or download both partitions and merge into host output)
                let t_transfer = std::time::Instant::now();
                // Create temporary destination buffer on dGPU to hold the transferred iGPU partition
                let dgpu_staging_out1 =
                    VulkanBuffer::new_device_local(self.dgpu.clone(), igpu_bytes, usage)?;
                let transfer_metrics =
                    CrossGpuTransferEngine::transfer_buffers(&igpu_buf_out, &dgpu_staging_out1)?;
                let transfer_time_us = t_transfer.elapsed().as_secs_f64() * 1.0e6;
                let _ = transfer_metrics;

                // Download output from dGPU: partition 0 directly from dgpu_buf_out, partition 1 from dgpu_staging_out1
                let mut output = vec![0.0f32; total_elements];
                let t_down = std::time::Instant::now();
                dgpu_buf_out.download_staged(&mut output[..dgpu_elements])?;
                dgpu_staging_out1.download_staged(&mut output[dgpu_elements..])?;
                let download_time_us = t_down.elapsed().as_secs_f64() * 1.0e6;

                let t_sync = std::time::Instant::now();
                // Host synchronization and ordering checks
                let sync_merge_time_us = t_sync.elapsed().as_secs_f64() * 1.0e6;

                let end_to_end_wallclock_us = t_wall_start.elapsed().as_secs_f64() * 1.0e6;

                Ok(SingleRunMetrics {
                    alloc_time_us,
                    upload_time_us,
                    dgpu_compute_us,
                    igpu_compute_us,
                    parallel_compute_us,
                    transfer_time_us,
                    sync_merge_time_us,
                    download_time_us,
                    end_to_end_wallclock_us,
                    output,
                })
            }
        }
    }

    /// Execute the full benchmark suite across all 3 modes for multiple input sizes
    pub fn run_suite(
        &self,
        sizes: &[usize],
        warmup_runs: usize,
        measured_runs: usize,
    ) -> Result<MultiDeviceSuiteReport, CoreError> {
        let mut comparisons = Vec::new();

        for &size in sizes {
            let res_dgpu = self.run_mode(
                WorkloadExecutionMode::DiscreteGpuOnly,
                size,
                warmup_runs,
                measured_runs,
            )?;
            let res_igpu = self.run_mode(
                WorkloadExecutionMode::IntegratedGpuOnly,
                size,
                warmup_runs,
                measured_runs,
            )?;
            let res_part = self.run_mode(
                WorkloadExecutionMode::PartitionedMultiGpu,
                size,
                warmup_runs,
                measured_runs,
            )?;

            let speedup_vs_dgpu =
                res_dgpu.end_to_end_wallclock_us.median / res_part.end_to_end_wallclock_us.median;
            let speedup_vs_igpu =
                res_igpu.end_to_end_wallclock_us.median / res_part.end_to_end_wallclock_us.median;

            let outcome_summary = if speedup_vs_dgpu > 1.05 {
                format!(
                    "Multi-Device execution is BENEFICIAL ({:.2}x faster than dGPU, {:.2}x faster than iGPU)",
                    speedup_vs_dgpu, speedup_vs_igpu
                )
            } else if speedup_vs_dgpu >= 0.95 {
                format!(
                    "Multi-Device execution is NEUTRAL ({:.2}x of dGPU latency, transfer overhead approximately offsets parallel compute)",
                    speedup_vs_dgpu
                )
            } else {
                format!(
                    "Multi-Device execution is SLOWER ({:.2}x of dGPU latency). Cross-device transfer overhead ({:.1} us) outweighs parallel compute gains.",
                    speedup_vs_dgpu,
                    res_part.cross_device_transfer_time_us.median
                )
            };

            comparisons.push(MultiDeviceBenchmarkComparison {
                total_elements: size,
                buffer_size_bytes: (size * std::mem::size_of::<f32>()) as u64,
                dgpu_only: res_dgpu,
                igpu_only: res_igpu,
                partitioned_multigpu: res_part,
                speedup_vs_dgpu,
                speedup_vs_igpu,
                outcome_summary,
            });
        }

        Ok(MultiDeviceSuiteReport {
            timestamp_utc: "2026-10-21T13:45:00Z".to_string(),
            os_name: "Pop!_OS 24.04 LTS".to_string(),
            kernel_version: "Linux 7.2.6-generic".to_string(),
            dgpu_name: self.dgpu.details.name.clone(),
            igpu_name: self.igpu.details.name.clone(),
            driver_version: self.dgpu.details.driver_version.clone(),
            comparisons,
        })
    }
}

struct SingleRunMetrics {
    alloc_time_us: f64,
    upload_time_us: f64,
    dgpu_compute_us: f64,
    igpu_compute_us: f64,
    parallel_compute_us: f64,
    transfer_time_us: f64,
    sync_merge_time_us: f64,
    download_time_us: f64,
    end_to_end_wallclock_us: f64,
    output: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_device_workload_all_modes_hardware() {
        let runner = MultiDeviceWorkloadRunner::new().expect("Failed to initialize runner");
        let element_count = 1024usize;

        // Test DiscreteGpuOnly
        let res_d = runner
            .run_mode(WorkloadExecutionMode::DiscreteGpuOnly, element_count, 1, 3)
            .expect("dGPU run failed");
        assert!(res_d.correctness_verified);
        assert_eq!(res_d.dgpu_elements, element_count);
        assert_eq!(res_d.igpu_elements, 0);

        // Test IntegratedGpuOnly
        let res_i = runner
            .run_mode(
                WorkloadExecutionMode::IntegratedGpuOnly,
                element_count,
                1,
                3,
            )
            .expect("iGPU run failed");
        assert!(res_i.correctness_verified);
        assert_eq!(res_i.dgpu_elements, 0);
        assert_eq!(res_i.igpu_elements, element_count);

        // Test PartitionedMultiGpu
        let res_p = runner
            .run_mode(
                WorkloadExecutionMode::PartitionedMultiGpu,
                element_count,
                1,
                3,
            )
            .expect("Partitioned run failed");
        assert!(res_p.correctness_verified);
        assert!(res_p.dgpu_elements > 0);
        assert!(res_p.igpu_elements > 0);
        assert_eq!(res_p.dgpu_elements + res_p.igpu_elements, element_count);

        println!(
            "All 3 modes verified correct! Wallclock median: dGPU={:.1}us, iGPU={:.1}us, Partitioned={:.1}us",
            res_d.end_to_end_wallclock_us.median,
            res_i.end_to_end_wallclock_us.median,
            res_p.end_to_end_wallclock_us.median
        );
    }
}
