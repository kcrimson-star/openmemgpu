# OpenMemGPU + VVRAM — Future Milestones & Technical Roadmap

This roadmap documents upcoming architectural milestones for **OpenMemGPU** and the **Virtual VRAM (VVRAM)** subsystem.

We actively encourage and welcome open-source contributions from GPU systems researchers, Linux/Windows kernel developers, graphics engineers, and AI runtime contributors.

---

## Roadmap Overview

| Milestone | Target Domain | Key Technologies | Status | Contribution Level |
| :--- | :--- | :--- | :---: | :---: |
| **Milestone 1–8** | AMD Linux / RADV Vulkan, CLI & GUI | Rust, Ash, Naga, DRM Sysfs, eframe | **Completed & Verified** | ✅ Production Ready |
| **Milestone 9** | Gaming & Steam/Proton Integration | Vulkan Implicit Layer, DXVK, Proton | **Upcoming / Open RFC** | 🚀 High Impact (Good First Project) |
| **Milestone 10** | Local AI & LLM Inference Runtime | `llama.cpp`, GGUF, Vulkan Compute, KV Cache | **Upcoming / Open RFC** | 🧠 High Impact (AI Systems) |
| **Milestone 11** | Windows Operating System Support | Direct3D 12, DXGI, WDDM 2.x/3.x, `windows-rs` | **Upcoming / Open RFC** | 🪟 Architecture Expansion |
| **Milestone 12** | NVIDIA GPU Support & CUDA Interop | CUDA UVM, `cuMemMap`, NVLink, Driver API | **Upcoming / Open RFC** | 🟢 Vendor Heterogeneity |
| **Milestone 13** | Intel Xe & Arc Graphics Support | Intel Level Zero, Vulkan ANV, OneAPI | **Upcoming / Open RFC** | 🔷 Multi-Vendor APU |
| **Milestone 14** | Distributed & CXL Memory Pooling | Compute Express Link (CXL 2.0/3.0), RDMA/RoCE | **Research Phase** | 🔬 Long-Term Vision |

---

## Milestone 9: Vulkan Layer for Transparent Game & App Expansion (`VK_LAYER_OPENMEMGPU`)

### Objective
Enable commercial games (running via Steam, Proton, DXVK, or native Linux) and desktop creative applications to transparently use the expanded Virtual VRAM pool without requiring source code modifications.

### Architecture & Approach
Implement a Vulkan Implicit Layer in pure Rust using `ash` and `vulkan-layer`:
1. **Memory Property Interception (`vkGetPhysicalDeviceMemoryProperties`)**:
   - Intercept driver query calls from DXVK/Proton.
   - Report the unified VVRAM pool capacity (e.g. 10.0 GB) instead of the raw dGPU dedicated heap size (6.0 GB).
2. **Allocation Overflow Interception (`vkAllocateMemory`)**:
   - Track active allocations on the discrete GPU's dedicated GDDR6 heap.
   - When the dedicated VRAM threshold (e.g. 85% capacity) is reached, transparently redirect new texture/buffer allocations to the pre-allocated host-visible iGPU memory pool.
3. **Steam Integration**:
   - Install layer manifest at `~/.local/share/vulkan/implicit_layer.d/openmemgpu.json`.
   - Activate via environment variable: `ENABLE_OPENMEMGPU=1 %command%` in Steam game launch options.

### How Contributors Can Help
- Implement `crates/vulkan-layer` skeleton wrapping `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr`.
- Test layer stability with popular titles (*The Last of Us Part I*, *Hogwarts Legacy*, *Cyberpunk 2077*).

---

## Milestone 10: Native Local AI & LLM Inference Runtime (`llama.cpp` Integration)

### Objective
Maximize performance and context length for local Large Language Models (LLMs) by partitioning transformer layer weights and KV cache memory across the discrete and integrated GPUs.

### Architecture & Approach
1. **Model Layer Slicing**:
   - Utilize existing [`crates/integrations/src/ai_model.rs`](file:///home/mike/Desktop/prjkt/openmemgpu/crates/integrations/src/ai_model.rs) layer-slicing logic.
   - Pin the majority of transformer blocks (e.g. layers 0–24 of an 8B model) in fast 6 GB GDDR6 on the dGPU.
   - Allocate the remaining blocks (layers 25–31) and large context KV caches in the expanded iGPU DDR4 memory pool.
2. **Pipelined Execution Coordination**:
   - While the dGPU computes layer $N$, asynchronously stream intermediate activations across the PCIe bus to the iGPU to compute layer $N+1$.
   - Minimize cross-device latency using Vulkan timeline semaphores and DMA-BUF zero-copy staging.
3. **Runtime Target**:
   - Create a backend plugin or integration with `llama.cpp` (`ggml-vulkan`) or an ONNX Runtime execution provider.

### How Contributors Can Help
- Benchmark token generation speed ($T/s$) of Llama 3 8B (Q4_K_M) across single-GPU vs. partitioned VVRAM.
- Implement KV cache auto-spilling into the iGPU pool when context length exceeds 4K tokens.

---

## Milestone 11: Windows OS Support (Direct3D 12, DXGI & WDDM)

### Objective
Bring OpenMemGPU's Virtual VRAM engine and desktop GUI to Windows 10 and Windows 11.

### Architecture & Approach
1. **Device Discovery & Telemetry via DXGI**:
   - Discover discrete and integrated adapters using Windows DXGI (`IDXGIFactory6::EnumAdapterByGpuPreference`).
   - Query live memory budgets and residency status via `IDXGIAdapter3::QueryVideoMemoryInfo`.
2. **Direct3D 12 Virtual Memory Management**:
   - Direct3D 12 natively supports reserved resources with virtual tile mappings (`ID3D12Device::CreateReservedResource`).
   - Map virtual GPU address ranges to physical memory pages spanning local dedicated VRAM and non-local shared system memory (`ID3D12CommandQueue::UpdateTileMappings`).
3. **Pure Rust Windows Implementation**:
   - Utilize Microsoft's official `windows-rs` crate for zero-cost Direct3D 12 and DXGI FFI bindings.
   - Implement `crates/backend-dx12` matching the existing `crates/backend-vulkan` trait interfaces.

### How Contributors Can Help
- Implement `crates/device-discovery/src/dxgi.rs` to enumerate adapters and query video memory budgets on Windows.
- Port the desktop GUI launcher for Windows (`openmemgpu.exe` and PowerShell scripts).

---

## Milestone 12: NVIDIA GPU Support & CUDA Interoperability

### Objective
Expand OpenMemGPU to support heterogeneous configurations featuring NVIDIA discrete GPUs (GeForce GTX/RTX, Quadro, Tesla) paired with AMD or Intel integrated GPUs.

### Architecture & Approach
1. **CUDA Unified Memory & Virtual Memory Management (`cuMemMap`)**:
   - Use CUDA Driver API virtual memory management (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`).
   - Map physical host-allocated memory pools directly into the CUDA virtual address space.
2. **Inter-Vendor Vulkan Coordination**:
   - Coordinate NVIDIA dGPU and AMD/Intel iGPU using cross-vendor Vulkan 1.3 `VK_KHR_external_memory` and timeline semaphores.
   - Probe NVLink / PCIe P2P direct transfer capabilities when multiple discrete cards are present.
3. **Telemetry & Monitoring**:
   - Interface with NVIDIA Management Library (`NVML` via `nvml-wrapper`) to poll temperature, power draw, and VRAM consumption in real time.

### How Contributors Can Help
- Implement `crates/backend-cuda` providing basic compute and buffer allocation interfaces.
- Test heterogeneous pairing: NVIDIA GeForce RTX (dGPU) + Intel Iris Xe / AMD Radeon (iGPU).

---

## Milestone 13: Intel Arc & Xe Graphics Integration

### Objective
Support Intel Arc discrete GPUs (Alchemist A770/A750, Battlemage) paired with Intel Core Ultra (Meteor Lake/Lunar Lake) Xe-LPG integrated GPUs.

### Architecture & Approach
1. **Vulkan Mesa ANV Support**:
   - Validate OpenMemGPU's buffer allocation and memory tiering logic against Mesa's Intel ANV Vulkan driver.
2. **Intel Level Zero API**:
   - Investigate Intel Level Zero API for fine-grained memory migration (`zeMemAllocHost`, `zeMemAllocDevice`, `zeMemAllocShared`).
3. **Hardware Resizable BAR**:
   - Exploit Intel Arc's mandatory Resizable BAR requirement for efficient large-chunk CPU-GPU transfers.

---

## Milestone 14: Compute Express Link (CXL) & Distributed Networked VVRAM

### Objective
Long-term systems research exploring disaggregated memory pooling where GPU VRAM is expanded using coherent host CXL 2.0/3.0 memory devices and remote node pooling via RoCE / InfiniBand RDMA.

---

## How to Contribute to a Milestone

1. **Check the Issues**: Browse existing milestone tracking issues at [github.com/kcrimson-star/openmemgpu/issues](https://github.com/kcrimson-star/openmemgpu/issues).
2. **Open an RFC / Discussion**: Before submitting large architectural changes, open an RFC issue outlining your proposed design, API changes, and verification plan.
3. **Follow the Rust Invariant**: All code authored for the project must be 100% Rust. External APIs must be accessed via audited FFI wrappers (`ash`, `windows-rs`, etc.).
4. **Benchmark & Verify**: Include reproducible benchmark numbers and tests verifying data correctness and memory safety.
