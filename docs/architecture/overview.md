# OpenMemGPU: System Architecture Overview

**Document Version**: 1.0  
**Status**: Architecture Baseline  

---

## 1. System Mission & Boundaries

OpenMemGPU coordinates computing devices, manages memory across physical boundaries, and schedules workloads for heterogeneous GPU systems.

```
+===================================================================================+
|                               COMPATIBLE WORKLOAD                                 |
|                  (e.g., Local AI / LLM Inference Engine, Compute)                 |
+===================================================================================+
                                         |
                                         v
+===================================================================================+
|                              OPENMEMGPU PUBLIC API                                |
|          Device Orchestration | VVRAM Allocation | Task Graph Submission          |
+===================================================================================+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +-------------------+           +--------------------+
|  VVRAM MANAGER   |           | WORKLOAD SCHEDULER|           | TRANSFER ENGINE    |
| (Residency,      |           | (Asymmetric load  |           | (DMA staging,      |
|  budget ledger,  |           |  partitioning,    |           |  dma-buf sharing,  |
|  eviction/paging)|           |  queue selection) |           |  timeline sync)    |
+------------------+           +-------------------+           +--------------------+
         |                               |                               |
         +-------------------------------+-------------------------------+
                                         |
                                         v
+===================================================================================+
|                                BACKEND SUBSYSTEM                                  |
+-----------------------------------------------------------------------------------+
|  BackendVulkan: Vulkan 1.4 compute, memory heaps, timeline semaphores, queues     |
|  BackendAmd:    Mesa RADV vendor telemetry, DRM sysfs memory profiling             |
|  BackendHost:   CPU multi-threaded compute fallback and staging manager           |
+===================================================================================+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +-------------------+           +--------------------+
| Discrete GPU     |           | Integrated GPU    |           | Host CPU & Memory  |
| Navi 10 (RDNA1)  |           | Renoir Vega 6     |           | AMD Ryzen 5 4600H  |
| 6.0 GiB GDDR6    |           | Shared System UMA |           | 14.0 GiB DDR4 RAM  |
+------------------+           +-------------------+           +--------------------+
```

---

## 2. Core Subsystems

### 2.1 Core Types & Orchestration (`crates/core`)
- Defines canonical representations of devices, capabilities, memory domains, and operational errors.
- Provides unified logging, telemetry hooks, and configuration structures.

### 2.2 Device Discovery (`crates/device-discovery`)
- Scans system bus and graphics drivers.
- Queries physical device properties, memory heaps, queue families, and vendor features.
- Distinguishes discrete GPUs, integrated GPUs, and CPU fallback engines.

### 2.3 Memory & VVRAM (`crates/memory` and `crates/vvram`)
- `crates/memory`: Manages the low-level physical memory domain descriptors, budget ledgers, and allocation handles.
- `crates/vvram`: Exposes the high-level virtual allocation interface, tracks dynamic residency, and executes eviction/prefetch policies.

### 2.4 Scheduler & Workload Partitioning (`crates/scheduler`)
- Partitions compute operations according to the measured throughput ratio of available devices.
- On our target platform (7.0 TFLOPS Navi 10 vs 1.1 TFLOPS Vega 6), the scheduler weights compute-heavy tasks heavily towards the dGPU (~85/15 ratio) or employs pipelined layer parallelism.

### 2.5 Transfer Subsystem (`crates/transfer`)
- Manages high-speed data migration across devices.
- Coordinates host-visible staging buffers and dma-buf file descriptor sharing.
- Enforces non-blocking execution using Vulkan timeline semaphores.

### 2.6 Workload Integrations (`crates/integrations`)
- Connects local AI/LLM runtimes to VVRAM and OpenMemGPU.
- Supports model tensor placement, layer offloading, and KV-cache paging.
