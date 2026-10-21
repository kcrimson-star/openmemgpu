# Hardware Inventory: OpenMemGPU & VVRAM Target Platform

**Document Version**: 1.0  
**Inspection Date**: October 21, 2026  
**Methodology**: Direct inspection via Linux sysfs (`/sys/class/drm`), Vulkan 1.4 API query (`vulkaninfo`), and PCI bus configuration (`lspci`).

---

## 1. Executive Summary

The target platform is a dual-GPU AMD mobile computing system featuring an integrated GPU (iGPU) connected to main system RAM and a discrete GPU (dGPU) equipped with dedicated GDDR6 VRAM and Resizable BAR (Smart Access Memory). Both GPUs run concurrently under the open-source AMD Linux kernel driver (`amdgpu`) and the Mesa RADV Vulkan 1.4 driver.

```
+----------------------------------------------------------------------------------------------------+
|                                         AMD HOST SYSTEM                                            |
|                                                                                                    |
|  +-------------------------------------+                +---------------------------------------+  |
|  |     Discrete GPU (dGPU): Navi 10    |                |       Integrated GPU (iGPU): Vega 6   |  |
|  |     AMD Radeon RX 5600M OEM         |                |       AMD Ryzen 5 4600H APU           |  |
|  |     PCI: 03:00.0 (AMDGPU / RADV)    |                |       PCI: 07:00.0 (AMDGPU / RADV)    |  |
|  |                                     |                |                                       |  |
|  |  +-------------------------------+  |                |  +---------------------------------+  |  |
|  |  | Dedicated VRAM (GDDR6)        |  |                |  | Pre-allocated Framebuffer       |  |  |
|  |  | Capacity: 5.98 GiB (6.0 GB)   |  |                |  | Capacity: 512 MiB               |  |  |
|  |  | Resizable BAR: 100% visible   |  |                |  | Used by Display: ~485 MiB       |  |  |
|  |  | Bandwidth: ~288 GB/s          |  |                |  | Status: Reserved for OS/Display |  |  |
|  |  +---------------+---------------+  |                |  +----------------+----------------+  |  |
|  |                  |                  |                |                   |                   |  |
|  |                  | GTT (PCIe DMA)   |                |                   | GTT (InfinityFab) |  |
|  +------------------+------------------+                +-------------------+-------------------+  |
|                     |                                                       |                      |
|                     |                  +------------------+                 |                      |
|                     +----------------->|  HOST SYSTEM RAM |<----------------+                      |
|                                        |  Total: 14.0 GiB |                                        |
|                                        |  Available: ~4.6G|                                        |
|                                        |  DDR4-3200 Dual  |                                        |
|                                        +------------------+                                        |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Device 0: Discrete GPU (AMD Radeon RX 5600M)

### 2.1 Hardware and PCI Identification
- **Product Name**: AMD Radeon RX 5600M OEM / 5600 XT
- **ASIC Family**: Navi 10 (RDNA 1.0 architecture)
- **PCI Device ID**: `1002:731f`
- **PCI Subsystem ID**: `1028:09f5` (Dell Inc.)
- **PCI Bus Topology**: `0000:03:00.0`
- **Display Nodes**: `card1-DP-1`, `card1-HDMI-A-1`
- **DRM Render Node**: `/dev/dri/renderD128`
- **Device UUID**: `00000000-0300-0000-0000-000000000000`
- **Driver UUID**: `414d442d-4d45-5341-2d44-525600000000`

### 2.2 Memory Domains and Heaps (Vulkan & Sysfs)
- **Heap 1 (Device-Local VRAM)**:
  - Total Size: `6,425,673,728` bytes (5.98 GiB / 6,128 MiB).
  - Flags: `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`.
  - Live Driver Budget: `6,368,702,464` bytes (5.93 GiB).
  - Current In-Use: `56,958,976` bytes (~54 MiB).
  - Remaining Available: `6,311,743,488` bytes (~5.88 GiB).
  - CPU Visibility (Resizable BAR / SAM): `6,425,673,728` bytes (100% of VRAM is directly CPU addressable).
- **Heap 0 (Host GTT / System Memory)**:
  - Total Size: `8,046,026,752` bytes (7.49 GiB / 7,673 MiB).
  - Flags: `0` (Non-device-local).
  - Live Driver Budget: `8,020,000,768` bytes (7.47 GiB).
  - Current In-Use: `26,025,984` bytes (~24 MiB).

### 2.3 Memory Types (Navi 10)
- **Type 0**: Heap 1 (`DEVICE_LOCAL`) — High-bandwidth VRAM for pure GPU compute.
- **Type 2**: Heap 0 (`HOST_VISIBLE | HOST_COHERENT`) — Host RAM mapped directly to GPU, uncached by CPU.
- **Type 3**: Heap 1 (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) — Resizable BAR VRAM: fast CPU writes without staging bounce!
- **Type 5**: Heap 0 (`HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`) — Host RAM cached by CPU, ideal for host readbacks and staging buffers.
- **Type 7**: Heap 1 (`DEVICE_LOCAL | DEVICE_COHERENT_AMD | DEVICE_UNCACHED_AMD`) — Low-latency atomic/signal buffers on GPU.

---

## 3. Device 1: Integrated GPU (AMD Radeon RX Vega 6 / Renoir)

### 3.1 Hardware and PCI Identification
- **Product Name**: AMD Radeon Graphics (Renoir APU, Vega 6)
- **ASIC Family**: Vega (5th Gen GCN architecture, gfx902/gfx90c)
- **PCI Device ID**: `1002:1636`
- **PCI Subsystem ID**: `1028:09f5` (Dell Inc.)
- **PCI Bus Topology**: `0000:07:00.0`
- **Display Nodes**: `card2-eDP-1` (drives internal 120Hz/60Hz laptop display)
- **DRM Render Node**: `/dev/dri/renderD129`
- **Device UUID**: `00000000-0700-0000-0000-000000000000`

### 3.2 Memory Domains and Heaps (Vulkan & Sysfs)
- **Heap 1 (Device-Local / Carved-out System Memory)**:
  - Vulkan Reported Size: `5,721,931,776` bytes (5.33 GiB).
  - Live Driver Budget: `4,870,258,688` bytes (4.54 GiB).
  - Dedicated Hardware Carveout (`mem_info_vram_total`): `536,870,912` bytes (512 MiB).
  - Framebuffer Memory Used by Desktop (`mem_info_vram_used`): `508,731,392` bytes (485 MiB).
- **Heap 0 (Host GTT / System Memory)**:
  - Vulkan Reported Size: `2,860,965,888` bytes (2.66 GiB).
  - Live Driver Budget: `2,435,133,440` bytes (2.27 GiB).
  - Sysfs GTT Total: `8,046,026,752` bytes (7.49 GiB).
  - Sysfs GTT Used: `771,387,392` bytes (~735 MiB).

### 3.3 Critical Technical Distinction for iGPU
Because the iGPU does not possess discrete memory chips, its "device-local" heap is allocated from main DDR4 system memory. Allocating from iGPU Heap 1 consumes the **same physical memory chips** as system RAM allocations. The OpenMemGPU resource tracker must account for this shared backing store to prevent oversubscription.

---

## 4. Compute Queues and Synchronization Features

Both GPUs offer asynchronous compute queues distinct from their universal graphics queues:
- **Queue Family 0**: Universal (Graphics + Compute + Transfer + Sparse Binding), count = 1.
- **Queue Family 1**: Async Compute (Compute + Transfer + Sparse Binding), count = 4, 64-bit nanosecond timestamp precision.
- **Queue Family 4**: SDMA / Sparse Binding engine.
- **Synchronization**: `VK_KHR_timeline_semaphore` fully supported across both devices, enabling non-blocking CPU-GPU dependency tracking and cross-queue event sequencing.
