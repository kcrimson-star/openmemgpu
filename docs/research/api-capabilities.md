# API Capabilities & Hardware Interface Analysis

**Document Version**: 1.0  
**Date**: October 21, 2026  
**Target Hardware**: AMD Navi 10 dGPU + AMD Renoir APU iGPU (Linux 7.2 / Mesa 26.1 RADV)  

---

## 1. Verified API Inventory

| API / Extension | Level | Supported on dGPU | Supported on iGPU | Primary Function in OpenMemGPU / VVRAM |
| :--- | :--- | :---: | :---: | :--- |
| **Vulkan Core 1.4** | Core | Yes | Yes | Baseline compute, dispatch, pipeline, and resource management. |
| **VK_EXT_memory_budget** | Ext | Yes | Yes | Live query of real driver memory budget and current usage per heap. |
| **VK_KHR_external_memory** | Ext | Yes | Yes | Defines capability to export and import memory across devices and processes. |
| **VK_KHR_external_memory_fd** | Ext | Yes | Yes | POSIX dma-buf export/import for Linux zero-copy buffer sharing. |
| **VK_KHR_timeline_semaphore** | Core 1.2 | Yes | Yes | Monotonically increasing 64-bit integer semaphores for multi-queue sync. |
| **VK_KHR_buffer_device_address** | Core 1.2 | Yes | Yes | GPU virtual memory pointers (64-bit) for pointer-chasing and BDA access. |
| **VK_KHR_device_group** | Core 1.1 | Yes | Yes | Grouped device execution (note: dGPU and iGPU form separate groups). |
| **Linux DRM Sysfs (`amdgpu`)** | OS | Yes | Yes | Live kernel driver counters (`mem_info_vram_total/used`, `gtt_total/used`). |

---

## 2. In-Depth Analysis of Key Mechanisms

### 2.1 Vulkan Memory Budget Query (`VK_EXT_memory_budget`)
- **Mechanism**: Attaching `VkPhysicalDeviceMemoryBudgetPropertiesEXT` to the `pNext` chain of `vkGetPhysicalDeviceMemoryProperties2`.
- **Returned Data**:
  - `heapBudget[VK_MAX_MEMORY_HEAPS]`: Live maximum memory the application can allocate from each heap without causing OS thrashing or eviction.
  - `heapUsage[VK_MAX_MEMORY_HEAPS]`: Current bytes allocated by the entire process across each heap.
- **Why it is essential for VVRAM**:
  Standard `vkGetPhysicalDeviceMemoryProperties` reports only static capacity (e.g. 5.98 GiB). It does not reflect other applications using VRAM (e.g. browsers, games, desktop compositors). `VK_EXT_memory_budget` provides real-time headroom, allowing VVRAM to make accurate allocation and placement decisions without guessing.

### 2.2 Cross-Device Data Movement: Export/Import vs Staging Buffers

VVRAM supports two verified data movement pathways between dGPU and iGPU:

```
Pathway A: Staging via Host-Visible Coherent Memory (Universal & Deterministic)
+-------------+      Host-to-Device Copy      +-------------+      Device-to-Host Copy      +-------------+
|  dGPU VRAM  | <===========================> |  Host RAM   | <===========================> |  iGPU GTT   |
| (Heap 1)    |    PCIe DMA (8-16 GB/s)       | (Staging)   |    Infinity Fabric (35 GB/s)  | (Heap 1/0)  |
+-------------+                               +-------------+                               +-------------+

Pathway B: Linux DRM dma-buf FD Export / Import (Zero-Copy Host Mapping)
+-------------+                                                                             +-------------+
|  dGPU VRAM  | ---[Export VkDeviceMemory as dma-buf fd]---> [Import into iGPU VkDevice] ---> |  iGPU GTT   |
| (Exportable)|                               (Linux Kernel TTM)                            | (Peer View) |
+-------------+                                                                             +-------------+
```

1. **Pathway A (Host Staging Buffer)**:
   - Allocates a buffer in `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` (or `HOST_CACHED`).
   - Copies from dGPU VRAM to Staging Buffer via Vulkan `vkCmdCopyBuffer`.
   - Copies from Staging Buffer to iGPU memory via iGPU `vkCmdCopyBuffer`.
   - *Advantage*: 100% reliable across all drivers, vendors, and OS platforms; simple lifetime management.
   - *Overhead*: One extra memory pass through host RAM.

2. **Pathway B (dma-buf External Memory)**:
   - Uses `VK_KHR_external_memory_fd` (`vkGetMemoryFdKHR` / `vkAllocateMemory` with `VkImportMemoryFdInfoKHR`).
   - Exports the allocation from one `VkDevice` as a Linux file descriptor.
   - Imports the file descriptor into the other `VkDevice`.
   - *Advantage*: Zero bounce copy in userspace.
   - *Limitation*: Requires driver support for cross-device dma-buf sharing; PCIe root complex isolation on mobile platforms can restrict direct GPU access to remote VRAM, causing kernel fallback to system RAM.

### 2.3 Fine-Grained Synchronization (`VK_KHR_timeline_semaphore`)
- **Mechanism**: Unlike binary Vulkan semaphores that cycle between signaled and unsignaled states, timeline semaphores maintain a 64-bit unsigned integer value.
- **Capabilities**:
  - `vkWaitSemaphores`: Host CPU thread can wait for a specific point in GPU execution without idling the entire device.
  - `vkSignalSemaphore`: Host CPU can signal a GPU queue to resume execution.
  - Cross-Queue Synchronization: Queue submissions on the dGPU can wait for a timeline value signaled by a submission on the iGPU, enabling lock-free dependency graphs across heterogeneous GPUs.

### 2.4 Device Groups vs Independent `VkDevice` Instances
- **Investigation Finding**: `vulkaninfo` reveals that on heterogeneous AMD platforms (Navi 10 + Renoir APU), the driver enumerates each GPU in its **own individual device group** of size 1.
- **Architectural Consequence**:
  A single `VkDevice` cannot span both the Navi 10 and Renoir GPUs simultaneously. OpenMemGPU must initialize **two distinct `VkDevice` handles** (one for dGPU, one for iGPU), managing coordination, resource allocation, and synchronization explicitly in the software orchestration layer.

---

## 3. Linux DRM Kernel Driver Metrics (`/sys/class/drm`)

The Linux `amdgpu` kernel driver exposes direct, atomic sysfs telemetry nodes:
- `mem_info_vram_total`: Total physical dedicated VRAM bytes.
- `mem_info_vram_used`: Current driver VRAM usage.
- `mem_info_gtt_total`: Maximum GTT (system memory mapped to GPU) bytes.
- `mem_info_gtt_used`: Current GTT allocations.
- `mem_info_vis_vram_total`: Total CPU-visible VRAM (Resizable BAR).
- `mem_info_vis_vram_used`: Current CPU-visible VRAM usage.

**Integration Decision**: The OpenMemGPU telemetry module will poll these sysfs files as ground-truth kernel metrics to validate Vulkan's `VK_EXT_memory_budget` reports.
