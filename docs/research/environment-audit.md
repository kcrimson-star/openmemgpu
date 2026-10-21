# Environment Audit: OpenMemGPU & VVRAM Development Environment

**Audit Date**: October 21, 2026  
**Auditor**: Lead Systems Architect & GPU Systems Researcher (OpenMemGPU Project)  
**Host System**: AMD Advantage Mobile Platform (Dell G5 15 SE / Navi 10 + Renoir APU)  

---

## 1. Operating System and Kernel

| Component | Detected Specification | Verification Method | Notes |
| :--- | :--- | :--- | :--- |
| **OS Distribution** | Pop!_OS 24.04 LTS (Ubuntu 24.04 `noble` base) | `/etc/os-release` | System76 COSMIC desktop environment |
| **Kernel Version** | Linux 7.2.6-76070206-generic | `uname -r` | SMP PREEMPT_DYNAMIC x86_64 |
| **Desktop Session** | COSMIC (Wayland) | `echo $XDG_SESSION_TYPE` | Direct scanout disabled (`COSMIC_DISABLE_DIRECT_SCANOUT=1`) |
| **Graphics Subsystem** | Linux DRM/KMS (`amdgpu` driver module) | `ls -l /sys/class/drm` | Kernel driver in use for both iGPU and dGPU |

---

## 2. CPU Architecture and System Memory

| Component | Specification | Details |
| :--- | :--- | :--- |
| **Processor** | AMD Ryzen 5 4600H with Radeon Graphics | Zen 2 architecture (Renoir family 23, model 96, stepping 1) |
| **Cores / Threads** | 6 physical cores, 12 logical threads | 1 socket, 2 threads per core |
| **Clock Frequencies** | Base 3.0 GHz, Max Boost 4.0 GHz | Hardware pstate scaling active |
| **CPU Caches** | L1d 192 KiB, L1i 192 KiB, L2 3 MiB, L3 8 MiB | 2 CCX structure with 4 MiB L3 per CCX |
| **System RAM** | 15.1 GB total (~14.0 GiB available to Linux) | Dual-channel DDR4-3200 SO-DIMM (shared with APU iGPU) |
| **Swap Space** | 21.0 GiB swap partition | Active on NVMe/SSD |
| **Memory Pressure** | ~4.6 GiB available at baseline | 10 GiB active/cached by OS desktop and running tasks |

---

## 3. GPU Hardware Inventory

The system features an all-AMD heterogeneous asymmetric GPU configuration:

### 3.1 Discrete GPU (dGPU)
- **Model**: AMD Radeon RX 5600M OEM / 5600 XT (Navi 10 architecture, RDNA 1.0, gfx1010).
- **PCI Address**: `0000:03:00.0` (Subsystem: Dell `1028:09f5`, Device ID: `1002:731f`, Revision: `c2`).
- **Kernel Driver**: `amdgpu` kernel module.
- **Dedicated VRAM**: 6,128 MB (5.98 GiB reported by Vulkan, 6,425,673,728 bytes).
- **Resizable BAR (SAM)**: Fully enabled (`mem_info_vis_vram_total` = 6,425,673,728 bytes). All 6 GB of VRAM is directly CPU addressable.
- **Display Connections**: Connected to external outputs (`card1-DP-1`, `card1-HDMI-A-1`).

### 3.2 Integrated GPU (iGPU)
- **Model**: AMD Radeon RX Vega 6 (Renoir APU, GCN 5th gen, gfx902/gfx90c).
- **PCI Address**: `0000:07:00.0` (Subsystem: Dell `1028:09f5`, Device ID: `1002:1636`, Revision: `c7`).
- **Kernel Driver**: `amdgpu` kernel module.
- **Dedicated Framebuffer**: 512 MB pre-allocated carveout from system RAM (`mem_info_vram_total` = 536,870,912 bytes).
- **Display Connections**: Drives the internal laptop display panel (`card2-eDP-1`).
- **Shared Memory (GTT)**: Up to 7.49 GiB dynamically allocatable from system RAM.

---

## 4. GPU Driver and API Capabilities

### 4.1 Vulkan
- **Vulkan Loader**: `1.3.280` (`/usr/lib/x86_64-linux-gnu/libvulkan.so.1.3.280`).
- **Vulkan Driver**: Mesa RADV (`Mesa 26.1.6-1pop0~1787580452~24.04~a5619ea`).
- **Vulkan API Version Supported**: Vulkan 1.4.354 on both physical devices (Conformance version 1.4.0.0).
- **Key Vulkan Extensions Verified**:
  - `VK_EXT_memory_budget` (Active on both devices — provides live budget and usage queries).
  - `VK_KHR_external_memory` & `VK_KHR_external_memory_fd` (Linux dma-buf export/import for cross-GPU memory sharing).
  - `VK_KHR_timeline_semaphore` (Fine-grained host-GPU and GPU-GPU asynchronous synchronization).
  - `VK_KHR_buffer_device_address` (64-bit GPU virtual pointers for high-performance memory indexing).
  - `VK_KHR_device_group` & `VK_KHR_device_group_creation` (Multi-device coordination APIs).
- **Device Nodes**:
  - `/dev/dri/card1` & `/dev/dri/renderD128` -> dGPU (Navi 10).
  - `/dev/dri/card2` & `/dev/dri/renderD129` -> iGPU (Renoir).

### 4.2 ROCm / HIP
- **Status**: Not installed in standard system paths (`/opt/rocm` does not exist; `rocm-smi` and `hipcc` not found).
- **Architecture Support Analysis**: RDNA 1 (Navi 10) and Vega 6 Mobile have deprecated or community-only support in official ROCm releases (ROCm 6.x officially targets GFX908, GFX90A, GFX940, GFX1100). The user's environment variable `HSA_OVERRIDE_GFX_VERSION=10.3.0` indicates prior testing with ROCm gfx1030 emulation.
- **Decision**: Vulkan 1.4 (RADV) is the verified, high-performance, stable, vendor-supported compute and memory management backend for this hardware.

---

## 5. Development Tools and Toolchain Status

| Tool | Status | Version / Path | Required Action |
| :--- | :--- | :--- | :--- |
| **GCC** | Installed | `/usr/bin/gcc` | Usable for C runtime dependencies if needed |
| **Make** | Installed | `/usr/bin/make` | Available for auxiliary build scripts |
| **CMake** | Installed | `/usr/bin/cmake` | Available for native build tasks |
| **Python 3** | Installed | `/usr/bin/python3` (3.12) | Available for scripting and data verification |
| **Vulkan CLI** | Installed | `/usr/bin/vulkaninfo` | Available for hardware query verification |
| **Rustc / Cargo** | **Not Found** | Not in `$PATH` or `/home/mike/.cargo/bin` | **Action Required**: Install Rust toolchain via `rustup` in `~/.cargo` |

---

## 6. Access and Execution Environment Constraints

1. **Physical GPU Hardware Access**: Confirmed. Both `/dev/dri/renderD128` (dGPU) and `/dev/dri/renderD129` (iGPU) are accessible. `vulkaninfo` successfully communicates with the hardware.
2. **Standard Sandbox Mode**: Active in Antigravity agent shell. Blocks outbound network connections by default (prevents direct `curl` downloading without user approval / `BypassSandbox`).
3. **Workspace Location**: `/home/mike/Desktop/prjkt/openmemgpu`.
