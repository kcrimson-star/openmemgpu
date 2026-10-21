use openmemgpu_core::device::DeviceId;
use openmemgpu_vvram::PlacementPolicy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLayerPlacement {
    pub layer_index: usize,
    pub name: String,
    pub size_bytes: u64,
    pub target_device: DeviceId,
    pub policy: PlacementPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPlacementPlan {
    pub model_name: String,
    pub total_parameters: u64,
    pub total_size_bytes: u64,
    pub layers: Vec<ModelLayerPlacement>,
}

pub struct ModelSlicer;

impl ModelSlicer {
    /// Slices transformer model layers across dGPU (high compute layers), iGPU, and Host RAM.
    pub fn plan_placement(
        model_name: &str,
        num_layers: usize,
        bytes_per_layer: u64,
        dgpu_budget_bytes: u64,
        dgpu_id: DeviceId,
        igpu_id: DeviceId,
    ) -> ModelPlacementPlan {
        let max_dgpu_layers = (dgpu_budget_bytes / bytes_per_layer) as usize;
        let mut layers = Vec::new();

        for i in 0..num_layers {
            let (target_device, policy) = if i < max_dgpu_layers {
                (dgpu_id, PlacementPolicy::PreferDiscreteVram)
            } else {
                (igpu_id, PlacementPolicy::PreferApuMemory)
            };

            layers.push(ModelLayerPlacement {
                layer_index: i,
                name: format!("transformer_layer_{}", i),
                size_bytes: bytes_per_layer,
                target_device,
                policy,
            });
        }

        ModelPlacementPlan {
            model_name: model_name.to_string(),
            total_parameters: (num_layers as u64) * 250_000_000,
            total_size_bytes: (num_layers as u64) * bytes_per_layer,
            layers,
        }
    }
}
