use openmemgpu_core::error::CoreError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmdDrmMemoryMetrics {
    pub vram_total_bytes: u64,
    pub vram_used_bytes: u64,
    pub gtt_total_bytes: u64,
    pub gtt_used_bytes: u64,
    pub vis_vram_total_bytes: u64,
    pub vis_vram_used_bytes: u64,
}

pub struct AmdBackend;

impl AmdBackend {
    pub fn read_card_metrics(card_name: &str) -> Result<AmdDrmMemoryMetrics, CoreError> {
        let base = Path::new("/sys/class/drm").join(card_name).join("device");
        if !base.exists() {
            return Err(CoreError::DriverError(format!(
                "DRM path not found: {:?}",
                base
            )));
        }

        let read_u64 = |filename: &str| -> u64 {
            fs::read_to_string(base.join(filename))
                .unwrap_or_default()
                .trim()
                .parse::<u64>()
                .unwrap_or(0)
        };

        Ok(AmdDrmMemoryMetrics {
            vram_total_bytes: read_u64("mem_info_vram_total"),
            vram_used_bytes: read_u64("mem_info_vram_used"),
            gtt_total_bytes: read_u64("mem_info_gtt_total"),
            gtt_used_bytes: read_u64("mem_info_gtt_used"),
            vis_vram_total_bytes: read_u64("mem_info_vis_vram_total"),
            vis_vram_used_bytes: read_u64("mem_info_vis_vram_used"),
        })
    }
}
