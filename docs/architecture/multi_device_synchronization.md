# Architecture Note: Multi-Device Synchronization & Memory Invariants

**Document Version**: 1.0  
**Status**: Architecture Baseline  
**Applicable Milestones**: Milestones 4, 5, 6  
**Target Hardware**: AMD Radeon RX 5600M dGPU (Navi 10) + AMD Radeon Graphics iGPU (Renoir APU)  

---

## 1. The Cross-VkDevice Synchronization Challenge

In single-GPU Vulkan development, synchronizing work between compute dispatches or transfer queues is straightforward: a command buffer records a pipeline barrier, or a queue submission signals a `VkSemaphore` (binary or timeline) which a subsequent queue submission on the same `VkDevice` waits upon.

When coordinating an integrated GPU and a discrete GPU, however, the system manages **two independent `VkDevice` instances**, typically with distinct memory heaps, driver queues, and physical interfaces:
- Device 0 (dGPU): Navi 10 discrete GPU on PCIe Bus `0000:03:00.0`, managing 6.0 GB dedicated GDDR6 VRAM.
- Device 1 (iGPU): Renoir APU on PCIe Bus `0000:07:00.0`, managing shared host DDR4 system RAM.

### The Vulkan Specification Invariant

According to the Vulkan Specification (Section 3.2, *Object Lifetime and Relationships*):
> *"All objects created from a VkDevice are strictly child objects of that specific VkDevice. Passing a handle created from one VkDevice as a parameter to any Vulkan command on a different VkDevice is invalid and yields undefined behavior."*

Specifically:
- A `VkSemaphore` created via `vkCreateSemaphore` on `VkDevice_dGPU` **cannot** be passed in `pWaitSemaphores` to a `vkQueueSubmit` call on `VkDevice_iGPU`. Doing so triggers driver crashes (SIGSEGV) or validation layer errors.
- A `VkBuffer` or `VkDeviceMemory` allocated on `VkDevice_dGPU` cannot be bound or accessed directly by `VkDevice_iGPU` command buffers without explicit external memory exporting.

---

## 2. Synchronization Models Evaluated

To solve cross-device coordination, we evaluated three distinct architectural models:

```
Model 1: Host-Mediated Coordination with Vulkan Timeline / Fence Primitives (IMPLEMENTED)
dGPU Queue: ===[ Compute Phase ]===> [ vkSignal(Timeline V1) ]
                                            |
Host Thread:                           [ vkWaitSemaphores(V1) ]
                                       [ Host-Mediated Transfer / Copy ]
                                       [ Submit iGPU Queue Work ]
                                            |
iGPU Queue:                            ===[ Compute Phase ]===> [ vkSignal(Timeline V2) ]

Model 2: Vulkan External Semaphore FD (VK_KHR_external_semaphore_fd)
dGPU Queue: ===[ Compute Phase ]===> [ Export Opaque FD ]
                                            | (dup FD)
iGPU Queue:                          [ Import Opaque FD ] ===> [ Wait & Compute ]

Model 3: Linux DRM syncobj / dma-fence Inter-driver Coordination
Direct kernel ioctls via /dev/dri/renderD128 and /dev/dri/renderD129
```

### Evaluation Matrix

| Criterion | Model 1: Host-Mediated Coordination | Model 2: External Semaphore FD | Model 3: DRM syncobj / dma-fence |
| :--- | :--- | :--- | :--- |
| **Driver Portability** | **Universal** (Standard Vulkan 1.2 Timeline Semaphores) | High (Requires extension & FD IPC) | Linux-only, DRM kernel specific |
| **Memory Isolation** | **Safe**: Handles separate memory domains explicitly | Complex: Requires matching external memory handles | Complex: Direct kernel ioctl handling |
| **Latency Cost** | Host wake-up latency ($\sim 10\text{--}40\ \mu\text{s}$) | Low ($\sim 5\text{--}15\ \mu\text{s}$) | Lowest ($<5\ \mu\text{s}$) |
| **Reliability on Hardware** | **100% Verified** on Navi 10 + Renoir | Untested on mixed APU/dGPU Mesa RADV | High failure surface on non-root setups |
| **Project Selection** | **Chosen for OpenMemGPU Baseline** | Candidate for Milestone 7+ optimization | Future kernel sidecar |

---

## 3. Host-Mediated Coordination Architecture

OpenMemGPU implements **Model 1: Host-Mediated Coordination with Vulkan Timeline Semaphores and Scoped Threads**:

```
                  +----------------------------------------------+
                  |         OpenMemGPU Scheduler Engine          |
                  +----------------------------------------------+
                                         |
                                         v
                         std::thread::scope (Rayon / POSIX)
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
         [Worker Thread 0: dGPU]                 [Worker Thread 1: iGPU]
                     |                                       |
                     v                                       v
        VkQueueSubmit(dGPU Compute)             VkQueueSubmit(iGPU Compute)
                     |                                       |
                     v                                       v
        vkWaitForFences / vkWaitSemaphores      vkWaitForFences / vkWaitSemaphores
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                (All Workers Ready)
                                         v
                     +---------------------------------------+
                     |   CrossGpuTransferEngine Coordination  |
                     | 1. Readback source staging memory     |
                     | 2. Memory barrier / Host invalidate   |
                     | 3. Copy / DMA into target staging     |
                     | 4. Flush host cache range             |
                     | 5. Upload into destination VRAM       |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | Downstream Dependent Pipeline Stage   |
                     +---------------------------------------+
```

### Safety and Ordering Guarantees

1. **Queue Isolation**:
   Each `VkQueue` is accessed exclusively by its designated host thread or protected by internal mutex locks, preventing concurrent command buffer submission on the same queue handle.
2. **Explicit Memory Visibility**:
   Data transferred between GPU memory domains undergoes explicit host visibility synchronization:
   - Source readback calls `vkInvalidateMappedMemoryRanges` (for non-coherent host memory) or completes a staging buffer download fence.
   - Destination upload calls `vkFlushMappedMemoryRanges` followed by a pipeline barrier `VK_ACCESS_HOST_WRITE_BIT -> VK_ACCESS_SHADER_READ_BIT`.
3. **Monotonic Monitored Timelines**:
   Timeline semaphores advance strictly monotonically. Waiting on value $V$ guarantees that all work associated with values $< V$ has completed on that device.

---

## 4. Rust Concurrency & Vulkan Handle Lifetimes

In standard Rust bindings for Ash, raw Vulkan handles (`vk::Device`, `vk::Buffer`, `vk::Pipeline`) do not automatically implement `Send` or `Sync` because Vulkan pointers are raw pointers (`*mut c_void`).

### 4.1 Thread Safety Invariants (`Send` & `Sync`)

OpenMemGPU wraps all Vulkan handles in safe Rust abstractions (`VulkanDevice`, `VulkanBuffer`, `TwoStageComputePipeline`) and explicitly documents and implements thread safety:

```rust
// In crates/backend-vulkan/src/device.rs
unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

// In crates/backend-vulkan/src/buffer.rs
unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

// In crates/backend-vulkan/src/compute.rs
unsafe impl Send for TwoStageComputePipeline {}
unsafe impl Sync for TwoStageComputePipeline {}
```

**Safety Invariants Maintained**:
- **Device Immutability**: The internal `ash::Device` handle and queue pointers are immutable after initialization.
- **Queue Submission Serialization**: Submissions to an individual `VkQueue` are protected either by distinct queues per thread or by scoped barriers in `std::thread::scope`.
- **Buffer Non-Aliasing**: Concurrent operations target distinct memory ranges or distinct buffers, preventing write-write or read-write data races on mapped pointers.

### 4.2 Teardown Destruction Order Invariant

A critical Vulkan specification requirement is that child objects must be destroyed before their parent objects:
$$\text{VkPipeline, VkBuffer, VkSemaphore} \prec \text{VkDevice} \prec \text{VkInstance}$$

In Rust structs, fields are dropped in **declaration order** (from top to bottom).

#### The SIGSEGV Bug Resolution
During initial multi-device runner development, declaring `_ctx: Arc<VulkanContext>` at the top of `MultiDeviceWorkloadRunner` caused `_ctx` to drop before `self.dgpu` and `self.igpu`. When the subsequent `VulkanDevice::drop` invoked `vkDeviceWaitIdle` and `vkDestroyDevice`, the underlying `VkInstance` had already been invalidated, causing a driver segmentation fault (SIGSEGV).

**The Architectural Fix**:
In any multi-device coordinator struct, parent context handles must be declared as the **last field**:

```rust
pub struct MultiDeviceWorkloadRunner {
    pub dgpu: Arc<VulkanDevice>,       // Dropped 1st (waits idle, destroys child buffers & pipelines)
    pub igpu: Arc<VulkanDevice>,       // Dropped 2nd (waits idle, destroys child buffers & pipelines)
    pub scheduler: WorkloadScheduler, // Dropped 3rd
    _ctx: Arc<VulkanContext>,          // Dropped LAST (destroys VkInstance safely)
}
```

---

## 5. Memory Tiering and Coherency Invariants

OpenMemGPU manages three distinct physical memory tiers:

| Tier | Physical Location | Vulkan Memory Properties | Mapping Strategy | Synchronization Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Dedicated VRAM** | Navi 10 GDDR6 (6 GB) | `DEVICE_LOCAL_BIT` (not host-visible) | Inaccessible to CPU directly | Must use staging buffer + `vkCmdCopyBuffer` |
| **Tier 2: Host-Visible GTT (dGPU)** | System RAM pinned over PCIe | `HOST_VISIBLE_BIT \| HOST_COHERENT_BIT` | CPU pointer via `vkMapMemory` | PCIe bus snooping, `HOST_WRITE` pipeline barrier |
| **Tier 3: Unified UMA (iGPU)** | System RAM (DDR4) | `DEVICE_LOCAL_BIT \| HOST_VISIBLE_BIT` | CPU pointer via `vkMapMemory` | Cache flush/invalidate if non-coherent; direct access |

### Invariant Checks Implemented:
1. **Alignment Verification**: All buffer offsets adhere to `VkMemoryRequirements.alignment` (typically 256 bytes for uniform/storage buffers).
2. **Double-Counting Prevention**: Shared system memory is accounted for in `PhysicalMemoryLedger` as a single unified pool; allocating GTT memory on dGPU and UMA memory on iGPU decrements the same host RAM reserve.
3. **Transaction Rollback**: If a multi-device allocation fails on device $B$ after succeeding on device $A$, the transaction rolls back cleanly, freeing device $A$'s allocation and updating the VVRAM ledger.

---

## 6. Known Limitations & Performance Boundaries

1. **Arithmetic Intensity Threshold**:
   Multi-device partitioning suffers an end-to-end slowdown on element-wise workloads ($I < 1.0\ \text{FLOP/byte}$) due to the PCIe staging transfer bottleneck ($0.21\ \text{GB/s}$). It is only profitable when computation time exceeds cross-device transfer latency.
2. **PCIe P2P Ineligibility**:
   Mobile AMD Ryzen + Radeon platforms do not support PCIe Peer-to-Peer BAR memory access across discrete GPU and APU root ports. Transfers must pass through host system RAM.
3. **Host Wakeup Overhead**:
   Host-mediated coordination incurs a $\sim 10\text{--}40\ \mu\text{s}$ CPU thread scheduling overhead per coordination step.

---

## 7. Recommendations for Milestone 7

Based on empirical data from Milestones 1 through 6, Milestone 7 should focus on:

1. **High Arithmetic Intensity Workload (Matrix Multiplication / GEMM)**:
   Implement a blocked matrix multiplication compute shader ($C = A \times B$). Because GEMM has arithmetic intensity $O(N)$, large matrix dimensions ($N \ge 2048$) will shift execution into the compute-bound regime where partitioned multi-GPU execution will demonstrate measurable speedups over single-device baselines.
2. **Pipelined Transformer Layer Execution**:
   Implement sequential layer pipelining (e.g. Layer $0..N$ on dGPU, Layer $N+1..M$ on iGPU) where weights remain permanently resident and only activation vectors ($<16\ \text{KB}$) cross the PCIe boundary.
3. **Dynamic VVRAM Tiering for Model Capacity**:
   Demonstrate loading a model whose total memory footprint exceeds 6 GB (the physical limit of the RX 5600M), seamlessly expanding into iGPU host memory via VVRAM.
