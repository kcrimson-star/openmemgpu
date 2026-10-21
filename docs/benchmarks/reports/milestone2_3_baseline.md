# Milestone 2–3 Benchmark Baseline: Vulkan Device Validation and Single-GPU Compute

**Benchmark Execution Date**: October 21, 2026  
**Evidence Standard**: `[Measured on Physical Hardware]`  
**Host Platform**: AMD Ryzen 5 4600H, Pop!_OS 24.04 LTS (Linux 7.2.6-generic)  
**Graphics Driver**: Mesa RADV 26.1.6 (Vulkan 1.4.354)  
**Machine-Readable Raw Data**: [`docs/benchmarks/results/milestone2_3_baseline.json`](file:///home/mike/Desktop/prjkt/openmemgpu/docs/benchmarks/results/milestone2_3_baseline.json)

---

## 1. Executive Summary

Milestones 2 and 3 establish the empirical hardware foundation for OpenMemGPU:
1. **Unambiguous Device Identification**: Both the discrete GPU (AMD Radeon RX 5600M Navi 10) and integrated GPU (AMD Radeon Vega 6 Renoir) are enumerated via Vulkan 1.4 and correlated with their Linux DRM render nodes (`/dev/dri/renderD128` and `renderD129`).
2. **Real GPU Compute Dispatches**: Verified physical execution of a parameterized vector multiplication compute shader (`C[i] = A[i] * B[i]`) on both GPUs independently with 100% numerical verification against CPU references.
3. **VVRAM Physical Materialization**: Connected `VvramManager` and `PhysicalMemoryLedger` to real `VulkanBuffer` allocations, verifying that physical ledger accounting tracks hardware allocations and reclaims memory on release.

---

## 2. Workload & Test Methodology

- **Workload**: Element-wise FP32 vector multiplication ($C[i] = A[i] \times B[i]$).
- **Data Size**: $1,048,576$ elements ($4,096$ KiB per buffer, 3 buffers = $12$ MiB total working memory).
- **Methodology**: 5 warm-up iterations followed by 20 consecutive measured steady-state iterations.
- **Shader Pipeline**: WGSL compute shader dynamically compiled to standard SPIR-V 1.3 via `naga v24.0.0` in pure Rust, configured to match native device wavefront size ($64$ threads).

---

## 3. Measured Physical Hardware Results

| Metric | AMD Radeon RX 5600M (Navi 10 dGPU) | AMD Radeon Graphics (Renoir APU iGPU) |
| :--- | :--- | :--- |
| **Device Kind** | Discrete GPU (`PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`) | Integrated GPU (`PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`) |
| **PCI Topology** | `0000:03:00.0` (`/dev/dri/renderD128`) | `0000:07:00.0` (`/dev/dri/renderD129`) |
| **Vulkan Driver Version** | Mesa RADV 26.1.6 (Vulkan 1.4.354) | Mesa RADV 26.1.6 (Vulkan 1.4.354) |
| **Reported Subgroup Size**| 64 threads (`Wave64` default in RADV) | 64 threads (`Wave64`) |
| **Live Device VRAM Budget**| $6,079$ MiB ($6,375,034,880$ bytes) | $5,172$ MiB ($5,424,005,120$ bytes) |
| **Correctness Check** | **VERIFIED (1,048,576 / 1,048,576 exact match)**| **VERIFIED (1,048,576 / 1,048,576 exact match)** |
| **Pipeline Init (one-time)**| $2,369.0\ \mu\text{s}$ | $2,155.0\ \mu\text{s}$ |
| **Buffer Alloc Median** | $3,567.1\ \mu\text{s}$ | $2,011.5\ \mu\text{s}$ |
| **Host Upload Median** | $2,275.9\ \mu\text{s}$ | $2,243.7\ \mu\text{s}$ |
| **Compute Dispatch Median**| $1,722.3\ \mu\text{s}$ | $726.1\ \mu\text{s}$ |
| **Device Readback Median** | $23,582.9\ \mu\text{s}$ | $17,674.9\ \mu\text{s}$ |
| **Total Steady-State** | $27,581.1\ \mu\text{s}$ | $20,644.8\ \mu\text{s}$ |
| **Effective Bandwidth** | **$7.31\ \text{GB/s}$** | **$17.33\ \text{GB/s}$** |

---

## 4. Key Architectural Insights & Findings

1. **Host-Visible Memory vs PCIe Bus Bottleneck**:
   - For host-visible buffers, the APU iGPU achieved **$17.33\ \text{GB/s}$** effective bandwidth compared to **$7.31\ \text{GB/s}$** on the discrete GPU.
   - *Explanation*: Host-visible memory on an APU is backed by system DDR4 RAM and accessed over the on-chip AMD Infinity Fabric without leaving the CPU package. Host-visible memory on the discrete GPU must transit the PCIe 3.0/4.0 bus.
   - *VVRAM Placement Rule*: High-throughput compute tasks on the discrete GPU must strictly target `DEVICE_LOCAL_BIT` (pure GDDR6 VRAM, ~288 GB/s), while host-visible staging buffers should be reserved for cross-device migration.

2. **PCIe Readback Latency (Device-to-Host)**:
   - On both devices, device-to-host readback ($23.6\ \text{ms}$ on dGPU, $17.7\ \text{ms}$ on iGPU) represents >85% of total end-to-end pipeline latency.
   - *VVRAM Placement Rule*: Workloads must keep intermediate tensors and activations resident in GPU memory across consecutive transformer layers, eliminating frequent host CPU readbacks.
