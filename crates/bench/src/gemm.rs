use ash::vk;
use openmemgpu_backend_vulkan::buffer::VulkanBuffer;
use openmemgpu_backend_vulkan::compute::GemmPipeline;
use openmemgpu_backend_vulkan::context::VulkanContext;
use openmemgpu_backend_vulkan::device::VulkanDevice;
use openmemgpu_backend_vulkan::physical_device::PhysicalDeviceEnumerator;
use openmemgpu_core::device::DeviceKind;
use openmemgpu_core::error::CoreError;
use openmemgpu_transfer::CrossGpuTransferEngine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmModeResult {
    pub mode: String,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub compute_time_us: f64,
    pub transfer_time_us: f64,
    pub wallclock_time_us: f64,
    pub gflops: f64,
    pub correctness_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmBenchmarkComparison {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub total_ops_giga: f64,
    pub dgpu_only: GemmModeResult,
    pub igpu_only: GemmModeResult,
    pub partitioned_multigpu: GemmModeResult,
    pub speedup_vs_dgpu: f64,
    pub speedup_vs_igpu: f64,
}

pub struct GemmBenchmarkRunner {
    pub dgpu: Arc<VulkanDevice>,
    pub igpu: Arc<VulkanDevice>,
    pipeline_dgpu: GemmPipeline,
    pipeline_igpu: GemmPipeline,
    _ctx: Arc<VulkanContext>,
}

impl GemmBenchmarkRunner {
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

        let pipeline_dgpu = GemmPipeline::new(dgpu.clone())?;
        let pipeline_igpu = GemmPipeline::new(igpu.clone())?;

        Ok(Self {
            dgpu,
            igpu,
            pipeline_dgpu,
            pipeline_igpu,
            _ctx: ctx,
        })
    }

    /// Calculate reference CPU GEMM for verification (on sampled elements to remain fast)
    pub fn verify_gemm_cpu(
        input_a: &[f32],
        input_b: &[f32],
        output_c: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        // Sample up to 100 random or corner coordinates
        let step = (m.max(1) * n.max(1)) / 64;
        let step = step.max(1);

        for idx in (0..(m * n)).step_by(step) {
            let row = idx / n;
            let col = idx % n;
            let mut expected = 0.0f32;
            for k_i in 0..k {
                expected += input_a[row * k + k_i] * input_b[k_i * n + col];
            }
            let actual = output_c[idx];
            let diff = (actual - expected).abs();
            let rel = diff / (expected.abs().max(1e-4));
            if rel > 1e-2 && diff > 1e-3 {
                return false;
            }
        }
        true
    }

    /// Run single-mode GEMM
    pub fn run_comparison(
        &self,
        m: u32,
        k: u32,
        n: u32,
        runs: usize,
    ) -> Result<GemmBenchmarkComparison, CoreError> {
        let size_a = (m * k) as usize;
        let size_b = (k * n) as usize;
        let size_c = (m * n) as usize;
        let total_ops = 2.0 * (m as f64) * (k as f64) * (n as f64);
        let total_ops_giga = total_ops / 1.0e9;

        let input_a: Vec<f32> = (0..size_a).map(|i| ((i % 19) as f32) * 0.05).collect();
        let input_b: Vec<f32> = (0..size_b).map(|i| ((i % 23) as f32) * 0.05).collect();

        let bytes_a = (size_a * std::mem::size_of::<f32>()) as u64;
        let bytes_b = (size_b * std::mem::size_of::<f32>()) as u64;
        let bytes_c = (size_c * std::mem::size_of::<f32>()) as u64;
        let bytes_params = 16u64;

        let usage_storage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let usage_uniform =
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        // -------------------------------------------------------------
        // 1. Discrete GPU Only
        // -------------------------------------------------------------
        let mut dgpu_times = Vec::new();
        let mut dgpu_out = vec![0.0f32; size_c];
        {
            let buf_a = VulkanBuffer::new_device_local(self.dgpu.clone(), bytes_a, usage_storage)?;
            let buf_b = VulkanBuffer::new_device_local(self.dgpu.clone(), bytes_b, usage_storage)?;
            let buf_c = VulkanBuffer::new_device_local(self.dgpu.clone(), bytes_c, usage_storage)?;
            let buf_p =
                VulkanBuffer::new_host_visible(self.dgpu.clone(), bytes_params, usage_uniform)?;

            buf_a.upload_staged(&input_a)?;
            buf_b.upload_staged(&input_b)?;
            let params = [m, k, n, 0u32];
            buf_p.upload(&params)?;

            for _ in 0..runs {
                let t_comp = self
                    .pipeline_dgpu
                    .dispatch(&buf_a, &buf_b, &buf_c, &buf_p, m, k, n)?;
                dgpu_times.push(t_comp);
            }
            buf_c.download_staged(&mut dgpu_out)?;
        }
        dgpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let dgpu_med_us = dgpu_times[dgpu_times.len() / 2];
        let dgpu_gflops = (total_ops / (dgpu_med_us * 1.0e-6)) / 1.0e9;
        let dgpu_verified = Self::verify_gemm_cpu(
            &input_a, &input_b, &dgpu_out, m as usize, k as usize, n as usize,
        );

        // -------------------------------------------------------------
        // 2. Integrated GPU Only
        // -------------------------------------------------------------
        let mut igpu_times = Vec::new();
        let mut igpu_out = vec![0.0f32; size_c];
        {
            let buf_a = VulkanBuffer::new_host_visible(self.igpu.clone(), bytes_a, usage_storage)?;
            let buf_b = VulkanBuffer::new_host_visible(self.igpu.clone(), bytes_b, usage_storage)?;
            let buf_c = VulkanBuffer::new_host_visible(self.igpu.clone(), bytes_c, usage_storage)?;
            let buf_p =
                VulkanBuffer::new_host_visible(self.igpu.clone(), bytes_params, usage_uniform)?;

            buf_a.upload(&input_a)?;
            buf_b.upload(&input_b)?;
            let params = [m, k, n, 0u32];
            buf_p.upload(&params)?;

            for _ in 0..runs {
                let t_comp = self
                    .pipeline_igpu
                    .dispatch(&buf_a, &buf_b, &buf_c, &buf_p, m, k, n)?;
                igpu_times.push(t_comp);
            }
            buf_c.download(&mut igpu_out)?;
        }
        igpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let igpu_med_us = igpu_times[igpu_times.len() / 2];
        let igpu_gflops = (total_ops / (igpu_med_us * 1.0e-6)) / 1.0e9;
        let igpu_verified = Self::verify_gemm_cpu(
            &input_a, &input_b, &igpu_out, m as usize, k as usize, n as usize,
        );

        // -------------------------------------------------------------
        // 3. Partitioned Multi-GPU (Asymmetric Row Slicing: ~86% dGPU, ~14% iGPU)
        // -------------------------------------------------------------
        let m_dgpu = ((m as f32 * 0.864).round() as u32).max(1).min(m - 1);
        let m_igpu = m - m_dgpu;

        let bytes_a_dgpu = ((m_dgpu * k) as usize * std::mem::size_of::<f32>()) as u64;
        let bytes_a_igpu = ((m_igpu * k) as usize * std::mem::size_of::<f32>()) as u64;
        let bytes_c_dgpu = ((m_dgpu * n) as usize * std::mem::size_of::<f32>()) as u64;
        let bytes_c_igpu = ((m_igpu * n) as usize * std::mem::size_of::<f32>()) as u64;

        let mut part_wallclock_times = Vec::new();
        let mut part_compute_times = Vec::new();
        let mut part_transfer_times = Vec::new();
        let mut part_out = vec![0.0f32; size_c];

        {
            // dGPU resources
            let buf_a_d =
                VulkanBuffer::new_device_local(self.dgpu.clone(), bytes_a_dgpu, usage_storage)?;
            let buf_b_d =
                VulkanBuffer::new_device_local(self.dgpu.clone(), bytes_b, usage_storage)?;
            let buf_c_d =
                VulkanBuffer::new_device_local(self.dgpu.clone(), bytes_c_dgpu, usage_storage)?;
            let buf_p_d =
                VulkanBuffer::new_host_visible(self.dgpu.clone(), bytes_params, usage_uniform)?;

            // iGPU resources
            let buf_a_i =
                VulkanBuffer::new_host_visible(self.igpu.clone(), bytes_a_igpu, usage_storage)?;
            let buf_b_i =
                VulkanBuffer::new_host_visible(self.igpu.clone(), bytes_b, usage_storage)?;
            let buf_c_i =
                VulkanBuffer::new_host_visible(self.igpu.clone(), bytes_c_igpu, usage_storage)?;
            let buf_p_i =
                VulkanBuffer::new_host_visible(self.igpu.clone(), bytes_params, usage_uniform)?;

            let split_a = (m_dgpu * k) as usize;
            buf_a_d.upload_staged(&input_a[..split_a])?;
            buf_b_d.upload_staged(&input_b)?;
            buf_p_d.upload(&[m_dgpu, k, n, 0u32])?;

            buf_a_i.upload(&input_a[split_a..])?;
            buf_b_i.upload(&input_b)?;
            buf_p_i.upload(&[m_igpu, k, n, 0u32])?;

            // Destination staging buffer on dGPU to hold the transferred iGPU C slice
            let dgpu_c_staging =
                VulkanBuffer::new_device_local(self.dgpu.clone(), bytes_c_igpu, usage_storage)?;

            for _ in 0..runs {
                let t_wall_start = std::time::Instant::now();

                // Parallel compute phase
                let t_comp_start = std::time::Instant::now();
                let (d_res, i_res) = std::thread::scope(|s| {
                    let h_d = s.spawn(|| {
                        self.pipeline_dgpu
                            .dispatch(&buf_a_d, &buf_b_d, &buf_c_d, &buf_p_d, m_dgpu, k, n)
                    });
                    let h_i = s.spawn(|| {
                        self.pipeline_igpu
                            .dispatch(&buf_a_i, &buf_b_i, &buf_c_i, &buf_p_i, m_igpu, k, n)
                    });
                    (h_d.join().unwrap(), h_i.join().unwrap())
                });
                let _ = d_res?;
                let _ = i_res?;
                let comp_dur_us = t_comp_start.elapsed().as_secs_f64() * 1.0e6;

                // Transfer iGPU result partition back to dGPU staging buffer
                let t_xfer_start = std::time::Instant::now();
                let _metrics = CrossGpuTransferEngine::transfer_buffers(&buf_c_i, &dgpu_c_staging)?;
                let xfer_dur_us = t_xfer_start.elapsed().as_secs_f64() * 1.0e6;

                let wall_dur_us = t_wall_start.elapsed().as_secs_f64() * 1.0e6;

                part_compute_times.push(comp_dur_us);
                part_transfer_times.push(xfer_dur_us);
                part_wallclock_times.push(wall_dur_us);
            }

            // Download dGPU partition and transferred partition into final output vector
            let split_c = (m_dgpu * n) as usize;
            buf_c_d.download_staged(&mut part_out[..split_c])?;
            dgpu_c_staging.download_staged(&mut part_out[split_c..])?;
        }

        part_wallclock_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        part_compute_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        part_transfer_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let part_wall_med_us = part_wallclock_times[part_wallclock_times.len() / 2];
        let part_comp_med_us = part_compute_times[part_compute_times.len() / 2];
        let part_xfer_med_us = part_transfer_times[part_transfer_times.len() / 2];

        let part_gflops = (total_ops / (part_wall_med_us * 1.0e-6)) / 1.0e9;
        let part_verified = Self::verify_gemm_cpu(
            &input_a, &input_b, &part_out, m as usize, k as usize, n as usize,
        );

        let speedup_vs_dgpu = dgpu_med_us / part_wall_med_us;
        let speedup_vs_igpu = igpu_med_us / part_wall_med_us;

        Ok(GemmBenchmarkComparison {
            m,
            k,
            n,
            total_ops_giga,
            dgpu_only: GemmModeResult {
                mode: "DiscreteGpuOnly".into(),
                m,
                k,
                n,
                compute_time_us: dgpu_med_us,
                transfer_time_us: 0.0,
                wallclock_time_us: dgpu_med_us,
                gflops: dgpu_gflops,
                correctness_verified: dgpu_verified,
            },
            igpu_only: GemmModeResult {
                mode: "IntegratedGpuOnly".into(),
                m,
                k,
                n,
                compute_time_us: igpu_med_us,
                transfer_time_us: 0.0,
                wallclock_time_us: igpu_med_us,
                gflops: igpu_gflops,
                correctness_verified: igpu_verified,
            },
            partitioned_multigpu: GemmModeResult {
                mode: "PartitionedMultiGpu".into(),
                m,
                k,
                n,
                compute_time_us: part_comp_med_us,
                transfer_time_us: part_xfer_med_us,
                wallclock_time_us: part_wall_med_us,
                gflops: part_gflops,
                correctness_verified: part_verified,
            },
            speedup_vs_dgpu,
            speedup_vs_igpu,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_benchmark_runner_hardware() {
        let runner =
            GemmBenchmarkRunner::new().expect("GemmBenchmarkRunner creation should succeed");
        // Fast test on 128x128x128
        let comp = runner
            .run_comparison(128, 128, 128, 3)
            .expect("GEMM comparison should succeed");
        assert!(comp.dgpu_only.correctness_verified);
        assert!(comp.igpu_only.correctness_verified);
        assert!(comp.partitioned_multigpu.correctness_verified);
        println!("GEMM 128x128 verified! dGPU: {:.1} us ({:.2} GFLOPS), iGPU: {:.1} us ({:.2} GFLOPS), Partitioned: {:.1} us ({:.2} GFLOPS)",
            comp.dgpu_only.wallclock_time_us, comp.dgpu_only.gflops,
            comp.igpu_only.wallclock_time_us, comp.igpu_only.gflops,
            comp.partitioned_multigpu.wallclock_time_us, comp.partitioned_multigpu.gflops
        );
    }
}
