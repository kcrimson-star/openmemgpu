# Known Limitations & Constraints: OpenMemGPU & VVRAM

**Document Version**: 1.0  
**Date**: October 21, 2026  
**Status**: Research & Audit Baseline  

---

## 1. Physical Memory Sharing & Double-Counting Hazards

1. **Shared Backing Store between Host RAM and iGPU**:
   - The integrated GPU (AMD Radeon RX Vega 6) does not possess physical on-die VRAM.
   - Its reported Vulkan "Device Local" heap (~5.33 GiB) and its Linux GTT heap (~7.49 GiB) are backed by the **same physical DDR4 system memory chips** (14.0 GiB total available to OS).
   - *Limitation*: A naive summation of dGPU VRAM (6 GB) + iGPU heaps (8 GB) + System RAM (14 GB) = 28 GB is completely illusory.
   - *VVRAM Mitigation Rule*: VVRAM must implement an explicit physical resource ledger (`PhysicalMemoryDomain`) that correlates physical RAM with iGPU allocations and dGPU GTT staging buffers, enforcing a global system memory ceiling.

2. **Display Compositor Contention on iGPU**:
   - The Wayland COSMIC desktop session runs on `card2-eDP-1` (the iGPU).
   - Of the 512 MiB dedicated carveout on the iGPU, **485 MiB (95%) is currently occupied by the display server**.
   - Any compute allocation on the iGPU will immediately overflow into system-memory-backed GTT.
   - Heavy compute workloads on the iGPU could trigger desktop frame drops or UI latency if memory bandwidth is saturated.

---

## 2. Bus Bandwidth & Heterogeneous Interconnects

| Path | Interconnect Type | Theoretical Bandwidth | Typical Measured Bandwidth | Latency Profile |
| :--- | :--- | :--- | :--- | :--- |
| **dGPU <-> VRAM** | GDDR6 192-bit bus | ~288 GB/s | ~240-260 GB/s | Nanoseconds (local) |
| **CPU <-> Host RAM** | Dual-channel DDR4-3200 | 51.2 GB/s | ~35-42 GB/s | ~60-80 ns |
| **iGPU <-> Host RAM** | AMD Infinity Fabric / UMA | ~51.2 GB/s | ~28-36 GB/s | Low (shared bus) |
| **dGPU <-> Host RAM** | PCIe 3.0/4.0 x8/x16 bus | 7.88 - 15.75 GB/s | ~6.5 - 12.0 GB/s | Microseconds (PCIe transit) |
| **dGPU <-> iGPU (Direct)** | No direct P2P link | N/A (Requires Host/DMA-buf) | Mediated via System RAM | High (bounce buffer) |

- *Limitation*: Direct PCIe peer-to-peer (P2P) transfers between an AMD APU iGPU and a mobile dGPU are not supported natively by PCIe root complex routing on mobile platforms.
- *VVRAM Mitigation Rule*: All data exchange between dGPU and iGPU must be planned as a **host-mediated or dma-buf export/import transfer**, utilizing CPU-visible host-coherent staging memory or Linux DRM dma-buf fd import.

---

## 3. Heterogeneous Compute Microarchitectures

1. **Subgroup / Wavefront Size Differences**:
   - Navi 10 (RDNA 1.0) executes native **Wave32** (32 threads per wavefront, with optional Wave64 mode).
   - Vega 6 (GCN 5.0) executes native **Wave64** (64 threads per wavefront).
   - *Limitation*: Compute shaders compiled with hardcoded workgroup sizes or subgroup assumptions optimized for Wave32 will suffer occupancy degradation on Vega 6, and vice versa.
   - *Mitigation*: The OpenMemGPU compute pipeline must query `VkPhysicalDeviceSubgroupProperties` and compile/dispatch shaders parameterized by the native wavefront width of each device.

2. **Compute Capability and Throughput Asymmetry**:
   - Navi 10 dGPU: ~7.0 TFLOPS FP32 compute power.
   - Vega 6 iGPU: ~1.1 TFLOPS FP32 compute power (~6.3x slower than dGPU).
   - *Limitation*: Naive 50/50 workload splitting will severely degrade overall throughput because the dGPU will stall waiting for the iGPU (Amdahl's law / straggler effect).
   - *OpenMemGPU Mitigation*: Workload partitioning must be **asymmetric and capability-weighted**, or pipelined (e.g. dGPU executes compute-heavy layers while iGPU handles embedding, sampling, or concurrent prefetching).

---

## 4. Software and Driver Limitations

1. **ROCm / HIP Support Absence**:
   - ROCm 6.x focuses on server/datacenter CDNA and select desktop RDNA 3 GPUs. Navi 10 (RDNA 1) and Vega Mobile lack official, stable ROCm 6.x packages.
   - *Decision*: OpenMemGPU's primary GPU backend must target **Vulkan 1.4 via Mesa RADV**. Vulkan is actively maintained, fully conforms to Vulkan 1.4, supports all required hardware queues, memory budgets, and cross-process sharing.

2. **Rust Toolchain Status**:
   - The Rust toolchain (`rustc`, `cargo`) is not currently installed on the host system.
   - Antigravity agent operates in Standard Sandbox Mode, blocking outbound network downloads by default.
   - *Requirement*: The user must approve toolchain installation in `~/.cargo` to proceed from Milestone 0 to Milestone 1.
