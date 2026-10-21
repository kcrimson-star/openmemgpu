# Prior Art Analysis: OpenMemGPU & VVRAM

**Document Version**: 1.0  
**Research Date**: October 21, 2026  
**Status**: Comprehensive Prior-Art Survey  

---

## 1. Executive Summary & Novelty Positioning

The goal of **OpenMemGPU** and **VVRAM** is to coordinate an integrated GPU (iGPU), a discrete GPU (dGPU), and system RAM into a software-managed memory and compute federation for compatible AI inference and compute workloads.

A rigorous evaluation of existing technologies demonstrates that while individual components (sub-allocators, multi-GPU splitters, unified memory drivers) exist, **no existing open-source system provides a unified, capability-aware, cross-device virtual memory coordinator across asymmetric AMD iGPU + dGPU architectures using standard cross-vendor graphics APIs (Vulkan)**.

```
+---------------------------------------------------------------------------------------------------+
| COMPARATIVE LANDSCAPE OF PRIOR ART                                                                |
+----------------------+--------------------+-------------------------------------------------------+
| Project / Technology | Primary Focus      | Key Limitations Compared to OpenMemGPU + VVRAM        |
+----------------------+--------------------+-------------------------------------------------------+
| Vulkan Memory Alloc  | Intra-device       | Allocates within a single VkDevice; no multi-GPU or   |
| (VMA / gpu-allocator)| sub-allocation     | cross-device migration logic.                         |
+----------------------+--------------------+-------------------------------------------------------+
| llama.cpp            | LLM inference      | Static layer-splitting (`-ngl`, `--split-mode`);     |
|                      |                    | no dynamic residency, paging, or memory budget awareness|
+----------------------+--------------------+-------------------------------------------------------+
| DeepSpeed / ZeRO-Off | PyTorch distributed| Server-grade CUDA/NVLink focus; requires homogeneous  |
|                      | training           | GPUs and massive CPU PCIe bandwidth; no Vulkan/APU.   |
+----------------------+--------------------+-------------------------------------------------------+
| vLLM                 | Serving throughput | PagedAttention manages KV-cache pages on a single GPU |
|                      |                    | or CUDA tensor-parallel nodes; no heterogeneous APU.  |
+----------------------+--------------------+-------------------------------------------------------+
| AMD SmartAccess Mem  | PCIe BAR mapping   | Exposes VRAM to CPU via PCIe; does not pool APU and   |
| (SAM / Resizable BAR)|                    | dGPU memory or provide software eviction/placement.   |
+----------------------+--------------------+-------------------------------------------------------+
| CUDA Unified Memory  | OS page-fault UVMM | Proprietary to NVIDIA; requires driver-level HMM;     |
|                      |                    | high page-fault thrashing under oversubscription.     |
+----------------------+--------------------+-------------------------------------------------------+
| OpenMemGPU + VVRAM   | Heterogeneous APU+ | Explicit software memory manager across Vulkan memory |
|                      | dGPU Virtual VRAM  | domains, budget-aware placement, async staging, and   |
|                      |                    | asymmetric compute coordination for local AI runtimes.|
+----------------------+--------------------+-------------------------------------------------------+
```

---

## 2. Detailed Survey of Related Technologies

### 2.1 Vulkan Sub-Allocators: VulkanMemoryAllocator (VMA) & `gpu-allocator`
- **What they do**: Provide sub-allocation of `VkDeviceMemory` blocks into small chunks, preventing exhaustion of the Vulkan driver's maximum allocation count (`maxMemoryAllocationCount`, typically 4,096).
- **Rust crates**: `gpu-allocator` (developed by Embark Studios), `vk-mem-rs` (safe bindings for AMD's C++ VMA).
- **Limitations**:
  - Bound to a single `VkDevice`.
  - Blind to other physical GPUs or system RAM pressure.
  - Do not track data replication, migration, or multi-device eviction.
- **OpenMemGPU Contribution**: VVRAM builds *above* device-level allocators. VVRAM manages logical addresses, domain placement, and cross-domain migration, delegating physical sub-allocation to device-specific allocators.

### 2.2 Inference Engines: `llama.cpp` and Layer Offloading
- **What they do**: Allows local LLM inference across CPU and GPUs. Supports Vulkan (`ggml-vulkan`), ROCm, and CUDA backends.
- **Memory Management Approach**:
  - Models are sliced horizontally by transformer layers.
  - Users manually specify how many layers run on GPU (`-ngl N`) and how layers are divided across multiple GPUs (`--split-mode row|layer`, `--tensor-split`).
- **Critical Limitations**:
  - **Static Placement**: If model layers do not fit into dedicated VRAM, loading crashes or aborts with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
  - **No Dynamic Migration**: Cannot dynamically swap layers in and out based on memory pressure or sequence length expansion.
  - **Identical Device Assumption**: Slicing logic assumes homogeneous compute throughput. Placing layers equally across a 7 TFLOPS dGPU and a 1 TFLOPS iGPU causes the dGPU to idle while waiting for the iGPU.
- **OpenMemGPU Contribution**: VVRAM provides an intelligent memory manager that allows layers to be staged dynamically, and OpenMemGPU schedules execution according to asymmetric compute throughput.

### 2.3 DeepSpeed ZeRO-Offload and ZeRO-Infinity
- **What they do**: Partitions model states across GPUs and offloads optimizer states and activations to host CPU memory or NVMe.
- **Limitations**:
  - Implemented inside Python / PyTorch / CUDA.
  - Designed for high-end server clusters with NVLink or PCIe 4.0 x16 links.
  - Incompatible with lightweight consumer laptops, embedded devices, and Vulkan compute.
- **OpenMemGPU Contribution**: A pure Rust, low-overhead systems implementation operable on standard consumer hardware without PyTorch or CUDA dependencies.

### 2.4 vLLM & PagedAttention
- **What they do**: Solves KV-cache fragmentation by allocating memory for Key and Value vectors in fixed-size pages (e.g. 16 tokens per block), analogous to OS virtual memory pages.
- **Limitations**:
  - Operates exclusively within a single GPU's address space or across identical CUDA GPUs in tensor-parallel configurations.
  - Does not manage model weights or heterogeneous memory domains (APU + dGPU + Host).
- **OpenMemGPU Contribution**: Adopts paging concepts from PagedAttention for KV-cache and weights, but extends the paging abstraction across heterogeneous memory domains (dedicated VRAM, APU shared memory, host DDR4).

### 2.5 Hardware and OS Facilities: Resizable BAR, SAM, and Linux DRM
- **Resizable BAR (AMD Smart Access Memory)**: Allows the CPU to map the entire GPU VRAM through a 64-bit PCIe BAR rather than the legacy 256 MiB window. Verified present on our Navi 10 dGPU (all 6.0 GB visible).
- **Linux DRM `amdgpu` driver**: Implements TTM (Translation Table Manager) for kernel-space GPU memory management, managing eviction between VRAM and GTT.
- **Vulkan External Memory (`VK_KHR_external_memory_fd`)**: Allows exporting a `VkDeviceMemory` allocation as a POSIX file descriptor (dma-buf) that can be imported by another device.
- **OpenMemGPU Contribution**: Uses Resizable BAR for zero-copy CPU writes into dGPU VRAM and leverages `VK_KHR_external_memory_fd` and staging buffers for deterministic cross-GPU data movement.

---

## 3. Rust GPU Ecosystem Review

| Crate | License | Status | Suitability for OpenMemGPU |
| :--- | :--- | :--- | :--- |
| `ash` | MIT / Apache 2.0 | Actively maintained, industry standard | **Recommended for Vulkan Backend**: Direct, zero-cost, type-safe Rust bindings to Vulkan 1.0 - 1.4. Provides complete control over physical devices, memory types, and extensions. |
| `gpu-allocator` | MIT / Apache 2.0 | Actively maintained (Embark) | Suitable for managing memory blocks within a specific `VkDevice`. |
| `wgpu` | MIT / Apache 2.0 | Actively maintained | High-level WebGPU abstraction. Abstraction hides Vulkan memory types, external memory fds, and device groups, making it unsuitable for low-level VVRAM control. |
| `vulkano` | MIT / Apache 2.0 | Maintained | Safe wrapper over Vulkan, but heavy compile times and opinionated abstraction layers compared to `ash`. |

---

## 4. Key References & Research Links

1. Khronos Group. *Vulkan 1.4 Specification: Memory Allocation & Memory Heaps*. (Accessed Oct 2026). https://registry.khronos.org/vulkan/specs/1.4/html/vkspec.html#memory
2. Sawicki, Adam (AMD). *Vulkan Memory Management Best Practices*. AMD Developer Blog. https://gpuopen.com/learn/vulkan-device-memory/
3. Gerdelan, Anton. *Vulkan Device Groups and Multi-GPU*. https://antongerdelan.net/opengl/vulkan/device_groups.html
4. Kwon, Woosuk et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), SOSP 2023. https://arxiv.org/abs/2309.06180
5. Rajbhandari, Samyam et al. *ZeRO-Offload: Democratizing Billion-Scale Model Training*, USENIX ATC 2021. https://arxiv.org/abs/2101.06840
6. Gerganov, Georgi. *llama.cpp: GGML Vulkan backend and memory architecture*. GitHub Repository.
