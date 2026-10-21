use openmemgpu_core::error::CoreError;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMemoryInfo {
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub os_reserved_ram_bytes: u64,
    pub dgpu_vram_bytes: u64,
    pub igpu_carveout_bytes: u64,
    pub igpu_gtt_bytes: u64,
    pub max_expandable_bytes: u64,
    pub recommended_steps_gb: Vec<u32>,
}

impl SystemMemoryInfo {
    pub fn discover() -> Result<Self, CoreError> {
        let meminfo = fs::read_to_string("/proc/meminfo").map_err(|e| {
            CoreError::DriverError(format!("Failed to read /proc/meminfo: {:?}", e))
        })?;

        let mut total_kb = 0u64;
        let mut avail_kb = 0u64;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    total_kb = parts[1].parse().unwrap_or(0);
                }
            } else if line.starts_with("MemAvailable:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    avail_kb = parts[1].parse().unwrap_or(0);
                }
            }
        }

        let total_ram_bytes = total_kb * 1024;
        let available_ram_bytes = avail_kb * 1024;

        // Default OS and Desktop environment reserve: 4 GB
        let os_reserved_ram_bytes = 4 * 1024 * 1024 * 1024u64;

        // Read dGPU and iGPU sysfs info if available
        let dgpu_vram_bytes = fs::read_to_string("/sys/class/drm/card1/device/mem_info_vram_total")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(6 * 1024 * 1024 * 1024);

        let igpu_carveout_bytes =
            fs::read_to_string("/sys/class/drm/card2/device/mem_info_vram_total")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(512 * 1024 * 1024);

        let igpu_gtt_bytes = fs::read_to_string("/sys/class/drm/card2/device/mem_info_gtt_total")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(8 * 1024 * 1024 * 1024);

        // Maximum expandable RAM that can be safely borrowed for VRAM:
        // total_ram minus OS reserve, capped at GTT limit
        let max_expandable_bytes = total_ram_bytes
            .saturating_sub(os_reserved_ram_bytes)
            .min(igpu_gtt_bytes);

        // Compute step options in GB (e.g. 0, 2, 4, 6, 8)
        let max_gb = (max_expandable_bytes / (1024 * 1024 * 1024)) as u32;
        let mut recommended_steps_gb = Vec::new();
        let mut step = 0u32;
        while step <= max_gb {
            recommended_steps_gb.push(step);
            step += 2;
        }
        if *recommended_steps_gb.last().unwrap_or(&0) != max_gb {
            recommended_steps_gb.push(max_gb);
        }

        Ok(Self {
            total_ram_bytes,
            available_ram_bytes,
            os_reserved_ram_bytes,
            dgpu_vram_bytes,
            igpu_carveout_bytes,
            igpu_gtt_bytes,
            max_expandable_bytes,
            recommended_steps_gb,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_memory_discovery() {
        let info = SystemMemoryInfo::discover().expect("Discovery should succeed on Linux");
        assert!(info.total_ram_bytes > 0);
        assert!(info.available_ram_bytes > 0);
        assert!(info.max_expandable_bytes > 0);
        assert!(!info.recommended_steps_gb.is_empty());
        println!(
            "Discovered System RAM: {} MB",
            info.total_ram_bytes / 1024 / 1024
        );
        println!(
            "Available for VVRAM Expansion: {} MB",
            info.max_expandable_bytes / 1024 / 1024
        );
        println!(
            "Recommended Expansion Steps: {:?}",
            info.recommended_steps_gb
        );
    }
}
