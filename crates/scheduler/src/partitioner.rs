use openmemgpu_core::device::DeviceId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadPartition {
    pub device_id: DeviceId,
    pub slice_start: usize,
    pub slice_count: usize,
    pub compute_weight: f32,
}

pub struct AsymmetricPartitioner;

impl AsymmetricPartitioner {
    /// Partitions total_elements across two asymmetric devices based on their compute TFLOPS.
    /// Prevents barrier idling by sizing workloads proportional to throughput.
    pub fn partition(
        dev_a: (DeviceId, f32),
        dev_b: (DeviceId, f32),
        total_elements: usize,
    ) -> (WorkloadPartition, WorkloadPartition) {
        let (id_a, tflops_a) = dev_a;
        let (id_b, tflops_b) = dev_b;

        let total_tflops = tflops_a + tflops_b;
        let weight_a = if total_tflops > 0.0 {
            tflops_a / total_tflops
        } else {
            0.5
        };
        let weight_b = 1.0 - weight_a;

        let count_a = ((total_elements as f32) * weight_a).round() as usize;
        let count_b = total_elements.saturating_sub(count_a);

        (
            WorkloadPartition {
                device_id: id_a,
                slice_start: 0,
                slice_count: count_a,
                compute_weight: weight_a,
            },
            WorkloadPartition {
                device_id: id_b,
                slice_start: count_a,
                slice_count: count_b,
                compute_weight: weight_b,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asymmetric_partitioning() {
        // Navi 10 (7.0 TFLOPS) vs Vega 6 (1.1 TFLOPS)
        let dgpu = (DeviceId(0), 7.0);
        let igpu = (DeviceId(1), 1.1);

        let (part_d, part_i) = AsymmetricPartitioner::partition(dgpu, igpu, 1000);

        // ~86% dGPU, ~14% iGPU
        assert_eq!(part_d.slice_count + part_i.slice_count, 1000);
        assert!(part_d.slice_count >= 860 && part_d.slice_count <= 870);
        assert!(part_i.slice_count >= 130 && part_i.slice_count <= 140);
    }
}
