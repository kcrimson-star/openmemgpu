use crate::partitioner::AsymmetricPartitioner;
use openmemgpu_core::device::DeviceId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadExecutionMode {
    DiscreteGpuOnly,
    IntegratedGpuOnly,
    PartitionedMultiGpu,
}

impl std::fmt::Display for WorkloadExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DiscreteGpuOnly => write!(f, "DiscreteGpuOnly"),
            Self::IntegratedGpuOnly => write!(f, "IntegratedGpuOnly"),
            Self::PartitionedMultiGpu => write!(f, "PartitionedMultiGpu"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingDecision {
    pub selected_mode: WorkloadExecutionMode,
    pub total_elements: usize,
    pub dgpu_slice_count: usize,
    pub igpu_slice_count: usize,
    pub dgpu_weight: f32,
    pub igpu_weight: f32,
    pub estimated_dgpu_time_us: f64,
    pub estimated_igpu_time_us: f64,
    pub estimated_partitioned_time_us: f64,
    pub estimated_transfer_overhead_us: f64,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadScheduler {
    pub min_partition_elements: usize,
    pub dgpu_tflops: f32,
    pub igpu_tflops: f32,
    pub cross_gpu_bandwidth_gbps: f64,
    pub cross_gpu_base_overhead_us: f64,
}

impl Default for WorkloadScheduler {
    fn default() -> Self {
        Self {
            // Below 256K elements (1 MB), cross-GPU coordination/transfer overhead dominates
            min_partition_elements: 262_144,
            dgpu_tflops: 7.0,
            igpu_tflops: 1.1,
            cross_gpu_bandwidth_gbps: 0.20,
            cross_gpu_base_overhead_us: 600.0,
        }
    }
}

impl WorkloadScheduler {
    pub fn new(min_partition_elements: usize) -> Self {
        Self {
            min_partition_elements,
            ..Default::default()
        }
    }

    /// Evaluates workload size, empirical hardware compute capability, and transfer costs
    /// to decide whether to execute on a single device or partition across dGPU and iGPU.
    pub fn decide(&self, total_elements: usize) -> SchedulingDecision {
        let (part_d, part_i) = AsymmetricPartitioner::partition(
            (DeviceId(0), self.dgpu_tflops),
            (DeviceId(1), self.igpu_tflops),
            total_elements,
        );

        let byte_size = (total_elements * std::mem::size_of::<f32>()) as f64;
        let igpu_bytes = (part_i.slice_count * std::mem::size_of::<f32>()) as f64;

        // Cost estimation based on empirical benchmarks:
        // dGPU: ~25 GB/s compute bandwidth + PCIe staging overhead
        let dgpu_compute_us = (byte_size / 25.0e9) * 1.0e6;
        let dgpu_upload_us = (byte_size * 2.0 / 1.5e9) * 1.0e6;
        let dgpu_download_us = (byte_size / 0.17e9) * 1.0e6;
        let estimated_dgpu_time_us = dgpu_upload_us + dgpu_compute_us + dgpu_download_us;

        // iGPU: ~18-30 GB/s compute bandwidth over Infinity Fabric UMA RAM
        let igpu_compute_us = (byte_size / 18.0e9) * 1.0e6;
        let igpu_upload_us = (byte_size * 2.0 / 2.0e9) * 1.0e6;
        let igpu_download_us = (byte_size / 0.17e9) * 1.0e6;
        let estimated_igpu_time_us = igpu_upload_us + igpu_compute_us + igpu_download_us;

        // Partitioned execution:
        // Max of parallel compute times + cross-GPU transfer of intermediate/final partition
        let part_dgpu_compute_us = ((byte_size * part_d.compute_weight as f64) / 25.0e9) * 1.0e6;
        let part_igpu_compute_us = ((byte_size * part_i.compute_weight as f64) / 18.0e9) * 1.0e6;
        let parallel_compute_us = part_dgpu_compute_us.max(part_igpu_compute_us);

        let estimated_transfer_overhead_us = self.cross_gpu_base_overhead_us
            + (igpu_bytes / (self.cross_gpu_bandwidth_gbps * 1.0e9)) * 1.0e6;

        let estimated_partitioned_time_us = dgpu_upload_us
            + parallel_compute_us
            + estimated_transfer_overhead_us
            + dgpu_download_us;

        if total_elements < self.min_partition_elements {
            SchedulingDecision {
                selected_mode: WorkloadExecutionMode::DiscreteGpuOnly,
                total_elements,
                dgpu_slice_count: part_d.slice_count,
                igpu_slice_count: part_i.slice_count,
                dgpu_weight: part_d.compute_weight,
                igpu_weight: part_i.compute_weight,
                estimated_dgpu_time_us,
                estimated_igpu_time_us,
                estimated_partitioned_time_us,
                estimated_transfer_overhead_us,
                rationale: format!(
                    "Workload size ({} elements, {} KB) is below minimum partitioning threshold ({} elements, {} KB). Single dGPU execution avoids cross-device transfer and coordination overhead.",
                    total_elements,
                    byte_size as usize / 1024,
                    self.min_partition_elements,
                    (self.min_partition_elements * 4) / 1024
                ),
            }
        } else if estimated_partitioned_time_us < estimated_dgpu_time_us {
            SchedulingDecision {
                selected_mode: WorkloadExecutionMode::PartitionedMultiGpu,
                total_elements,
                dgpu_slice_count: part_d.slice_count,
                igpu_slice_count: part_i.slice_count,
                dgpu_weight: part_d.compute_weight,
                igpu_weight: part_i.compute_weight,
                estimated_dgpu_time_us,
                estimated_igpu_time_us,
                estimated_partitioned_time_us,
                estimated_transfer_overhead_us,
                rationale: format!(
                    "Workload size ({} elements, {:.1} MB) is large enough to partition. Estimated speedup from dual-GPU compute ({:.1} us) offsets cross-GPU transfer overhead ({:.1} us).",
                    total_elements,
                    byte_size / (1024.0 * 1024.0),
                    parallel_compute_us,
                    estimated_transfer_overhead_us
                ),
            }
        } else {
            SchedulingDecision {
                selected_mode: WorkloadExecutionMode::DiscreteGpuOnly,
                total_elements,
                dgpu_slice_count: part_d.slice_count,
                igpu_slice_count: part_i.slice_count,
                dgpu_weight: part_d.compute_weight,
                igpu_weight: part_i.compute_weight,
                estimated_dgpu_time_us,
                estimated_igpu_time_us,
                estimated_partitioned_time_us,
                estimated_transfer_overhead_us,
                rationale: format!(
                    "Cross-GPU transfer and synchronization overhead ({:.1} us) outweighs parallel compute gains. Selecting DiscreteGpuOnly for lowest latency.",
                    estimated_transfer_overhead_us
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_threshold_enforcement() {
        let scheduler = WorkloadScheduler::default();

        // Small workload (16K elements = 64 KB): should select DiscreteGpuOnly
        let small_decision = scheduler.decide(16_384);
        assert_eq!(
            small_decision.selected_mode,
            WorkloadExecutionMode::DiscreteGpuOnly
        );
        assert!(small_decision.rationale.contains("below minimum"));

        // Large workload (16M elements = 64 MB): evaluates partitioned execution
        let large_decision = scheduler.decide(16_777_216);
        assert!(large_decision.dgpu_slice_count > 0);
        assert!(large_decision.igpu_slice_count > 0);
    }
}
