# Hardware & Runtime Compatibility Matrix

**Document Version**: 1.0  
**Updated**: October 21, 2026  

---

## 1. Supported GPU Architectures

| GPU Architecture | Vendor | Target Model Example | Backend | Status | Verified Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Navi 10 (RDNA 1.0)** | AMD | Radeon RX 5600M / 5600 XT | Vulkan (RADV) | **Verified** | 6GB VRAM, SAM / Resizable BAR, Async Compute, Timeline Semaphores |
| **Renoir (Vega GCN 5)** | AMD | Radeon Vega 6 (4600H APU) | Vulkan (RADV) | **Verified** | 512MB Framebuffer + 7.5GB GTT, Async Compute, Timeline Semaphores |
| **Navi 2x / 3x (RDNA 2/3)**| AMD | Radeon RX 6700M / 7800 XT | Vulkan (RADV/AMD) | Supported (Design) | Hardware ray-tracing, dual-issue compute |
| **Intel Xe / Arc** | Intel | Iris Xe / Arc A770 | Vulkan (ANV) | Planned (M9) | Multi-engine compute, unified driver |
| **NVIDIA Ada / Ampere** | NVIDIA | RTX 3060 / 4070 | Vulkan (NV) | Planned (M9) | Proprietary Vulkan extensions |

---

## 2. Operating System Support

| OS | Kernel / Build | Driver | Support Tier | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Pop!_OS 24.04 LTS** | Linux 7.2.6+ | Mesa 26.1.6 RADV | **Tier 1 (Reference)** | Primary development and validation platform. |
| **Ubuntu 24.04 LTS** | Linux 6.8 / 7.x | Mesa 24.x / 25.x RADV | **Tier 1** | Binary and API compatible with reference platform. |
| **Arch Linux / Fedora 40**| Linux 6.10+ | Mesa latest RADV | **Tier 1** | Bleeding edge kernel and driver support. |
| **Windows 11** | Build 22631+ | AMD Adrenalin 24.x+ | **Tier 2** | Supported via Vulkan 1.3 loader on Windows. |

---

## 3. Local AI / LLM Runtime Compatibility

| AI Runtime | Integration Mechanism | Status | Notes |
| :--- | :--- | :--- | :--- |
| **OpenMemGPU Native Tensor Engine** | Direct Rust Vulkan Shaders | **In Development (M7)** | Native matrix-multiply & attention kernels integrated directly with VVRAM buffers. |
| **llama.cpp (Vulkan backend)** | Custom allocator injection | **Planned (M7)** | Intercepting `ggml-vulkan` memory allocations to route through VVRAM. |
| **Candle (Hugging Face)** | Custom device backend | **Candidate (M7)** | Pure Rust ML framework with custom backend support. |
