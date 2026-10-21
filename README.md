# OpenMemGPU + VVRAM (Virtual VRAM)
### Autonomous Heterogeneous GPU Memory Orchestrator & Dynamic VRAM Expansion Runtime

[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Vulkan](https://img.shields.io/badge/vulkan-1.4%20conformance-red.svg)](https://www.vulkan.org/)
[![Hardware Verified](https://img.shields.io/badge/hardware-AMD%20Navi%2010%20%2B%20Renoir-brightgreen.svg)](#3-verified-physical-hardware-target)

**OpenMemGPU** is an open-source systems architecture, runtime engine, and user application for heterogeneous GPU platforms.  
Its primary innovation target is **VVRAM (Virtual VRAM)**: an intelligent, software-managed memory coordinator that dynamically pools a discrete GPU (dGPU), an integrated GPU (iGPU), and eligible system memory into a unified, high-capacity Virtual VRAM resource.

---

## 1. Key Features

- ⚡ **Dynamic VRAM Expansion Slider**:
  Increase effective GPU memory on the fly (`+2 GB`, `+4 GB`, `+6 GB`, etc.) by safely borrowing host-visible memory via the integrated GPU.
- 🧠 **Automatic System RAM Discovery & OS Reserve Protection**:
  Automatically discovers your PC's physical RAM, calculates live memory pressure, and enforces an inviolable $4.0\ \text{GB}$ OS/desktop reserve to prevent stuttering or out-of-memory errors.
- 🚀 **Hardware-Proven Oversized Buffer Allocations**:
  Allocate virtual buffers that exceed the physical dedicated VRAM boundary of a discrete GPU (e.g. $7.0\ \text{GB}$ or $8.0\ \text{GB}$ allocations on a $6.0\ \text{GB}$ dGPU), automatically partitioned across GDDR6 and host DDR4 memory.
- 🖥️ **Modern Minimalist Desktop GUI (`openmemgpu-gui`)**:
  Pure-Rust desktop application built with `eframe`/`egui` featuring an interactive expansion slider, live capacity gauges, real-time DRM/AMDGPU telemetry, one-click hardware benchmark runner, and diagnostic export.
- ⚙️ **High-Performance CLI (`openmemgpu-cli`)**:
  Feature-complete terminal interface for automation, continuous integration, headless servers, and benchmarking.
- 📊 **Empirical Multi-Device Execution & GEMM Engine**:
  High-arithmetic-intensity Matrix Multiplication (GEMM, $C = A \times B$) with tiled compute shaders delivering up to **$2.79\text{x}$ speedup** over integrated GPU execution.
- 🔒 **100% Pure Rust Architecture**:
  All project-owned core, memory, scheduler, transfer, benchmark, CLI, and GUI implementation code is 100% Rust without external C/C++ build dependencies.

---

## 2. Desktop Application & User Interface

OpenMemGPU includes a modern, minimalist desktop interface with dark slate styling, crisp typography, and real-time GPU telemetry.

### Running the Desktop GUI

You can launch OpenMemGPU with a single command:

```bash
# Using the launcher script (auto-detects Wayland / X11)
./scripts/launch-gui.sh

# Or directly via cargo
cargo run --release -p openmemgpu-gui
```

### Desktop Entry Integration (Linux)

To integrate OpenMemGPU into your Linux application launcher:

```bash
cp openmemgpu.desktop ~/.local/share/applications/
update-desktop-database ~/.local/share/applications/
```

### What You Can Do in the GUI

1. **Adjust the VRAM Slider**: Move the slider to select how many gigabytes to borrow from system memory (`+2 GB`, `+4 GB`, `+6 GB`, etc.).
2. **Apply Dynamic Allocation**: Click **"Apply Dynamic Allocation"** to provision real host-visible backing memory and update the physical ledger.
3. **Test Oversized Allocation**: Click **"Test Oversized Allocation (7.0 GB)"** to verify real-time multi-device buffer allocation exceeding the physical 6 GB dGPU limit.
4. **Run Live Hardware Benchmarks**: Switch to the **Benchmarks** tab, select GEMM or Vector workloads, and watch real-time concurrent dispatches across dGPU and iGPU queues.
5. **Inspect DRM Telemetry**: View real-time VRAM/GTT consumption, Resizable BAR visibility, and Vulkan driver budgets.

---

## 3. Verified Physical Hardware Target

OpenMemGPU is designed and verified against real heterogeneous consumer hardware:

- **Host Processor**: AMD Ryzen 5 4600H (6 cores / 12 threads, Renoir Zen 2) + 16.0 GB DDR4 RAM.
- **Discrete GPU**: AMD Radeon RX 5600M OEM (Navi 10 RDNA 1.0, 6.0 GB GDDR6, PCI `0000:03:00.0`, `/dev/dri/renderD128`).
- **Integrated GPU**: AMD Radeon Graphics Vega 6 (Renoir APU, 512 MB carveout + 7.5 GB shared DDR4, PCI `0000:07:00.0`, `/dev/dri/renderD129`).
- **Graphics Driver / OS**: Pop!_OS 24.04 LTS (Linux kernel `7.2.6-generic`), Mesa RADV 26.1.6, Vulkan 1.4.354.

---

## 4. Command-Line Interface (`openmemgpu-cli`)

For headless servers, scripts, or automated diagnostics, the CLI provides complete control:

### Discover Hardware & Memory Domains
```bash
openmemgpu-cli discover
```

### Inspect Live Status & Memory Budgets
```bash
openmemgpu-cli status
```

### Dynamic VRAM Capacity Expansion
```bash
# Query system memory discovery and safe expansion limits
openmemgpu-cli expand

# Dynamically provision +4 GB of Virtual VRAM (yielding 10.0 GB Unified Pool)
openmemgpu-cli expand --add-gb 4
```

### Run High-Intensity Matrix Multiplication (GEMM)
```bash
# Run 1024x1024 Matrix Multiplication across both GPUs
openmemgpu-cli gemm --size 1024 --runs 5 --output docs/benchmarks/results/milestone7_gemm_execution.json
```

### Run Multi-Device Compute Workload Suite
```bash
# Compare dGPU-only vs iGPU-only vs Partitioned Multi-GPU across 4 buffer sizes
openmemgpu-cli bench-multidevice --elements 16384,1048576,4194304,16777216 --runs 10
```

### Benchmark Memory Tiering (Dedicated GDDR6 vs Host GTT)
```bash
openmemgpu-cli bench-tiering --warmup 3 --runs 10
```

### Benchmark Cross-GPU Data Transfers
```bash
openmemgpu-cli bench-transfer --warmup 3 --runs 10
```

### Probe Linux DRM / Vulkan DMA-BUF Capabilities
```bash
openmemgpu-cli probe-dma-buf
```

---

## 5. Architecture & Workspace Structure

```
openmemgpu/
├── Cargo.toml
├── openmemgpu.desktop           # XDG Linux desktop application entry
├── scripts/
│   └── launch-gui.sh            # Desktop GUI launcher script
├── assets/
│   └── openmemgpu.svg           # High-resolution vector icon
├── crates/
│   ├── core/                    # Device kinds, error domains, memory types
│   ├── device-discovery/        # DRM sysfs & system memory auto-discovery
│   ├── backend-vulkan/          # Vulkan 1.4 compute, Naga SPIR-V, timeline semaphores
│   ├── backend-amd/             # AMDGPU DRM sysfs live telemetry parser
│   ├── memory/                  # Physical memory ledger & budget accounting
│   ├── vvram/                   # Virtual VRAM manager & dynamic capacity expander
│   ├── scheduler/               # Asymmetric compute partitioner & cost model
│   ├── transfer/                # Cross-GPU staging transfer engine & DMA probe
│   ├── bench/                   # GEMM and multi-device benchmark runners
│   ├── cli/                     # Diagnostic & benchmarking terminal CLI
│   ├── gui/                     # Modern minimalist desktop dashboard (egui/eframe)
│   └── integrations/            # Workload & model layer slicing adapters
└── docs/
    ├── architecture/            # Architecture notes & synchronization designs
    ├── benchmarks/reports/      # Detailed physical hardware benchmark reports
    └── benchmarks/results/      # Machine-readable benchmark JSON measurements
```

---

## 6. Milestone Progress & Empirical Evidence

Every milestone in OpenMemGPU is proven with physical hardware data:

- [Milestone 2–3 Report: Vulkan Baseline & Single-GPU Compute](docs/benchmarks/reports/milestone2_3_baseline.md)
  - Proved Vulkan 1.4 compute dispatch on Navi 10 and Renoir with pure-Rust Naga shader compilation.
- [Milestone 4–5 Report: Dedicated VRAM Tiering & Cross-GPU Transfers](docs/benchmarks/reports/milestone4_5_memory_tiering_and_transfers.md)
  - Proved dedicated GDDR6 delivers **$2.68\text{x}$ to $2.97\text{x}$ higher compute bandwidth** over GTT.
  - Proved bidirectional cross-GPU staging transfers with $100\%$ byte-for-byte data integrity.
- [Milestone 6 Report: Multi-Device Workload Execution](docs/benchmarks/reports/milestone6_multidevice_execution.md)
  - Demonstrated concurrent physical queue dispatches across dGPU and iGPU in scoped threads.
  - Identified the arithmetic intensity law and PCIe transfer latency boundary.
- [Milestone 7 Report: High-Arithmetic-Intensity GEMM & Dynamic Expansion](docs/benchmarks/reports/milestone7_high_arithmetic_intensity_gemm.md)
  - Proved GEMM compute scaling with **$2.79\text{x}$ speedup over iGPU** under $170.67\ \text{FLOP/byte}$.
  - Proved dynamic VRAM expansion and oversized $7.0\ \text{GB}$ virtual buffer allocations spanning physical devices.
- [Architecture Note: Multi-Device Synchronization Invariants](docs/architecture/multi_device_synchronization.md)
  - Detailed Vulkan `VkDevice` isolation, timeline semaphores, scoped threads, and teardown destruction order.

---

## 7. Future Milestones & Public Roadmap (Open for Community Contributions 🚀)

OpenMemGPU is designed from the ground up as a modular, vendor-extensible systems architecture. We actively welcome contributors from the open-source community to implement, test, and expand these planned milestones:

| Milestone | Target Domain | Key Architecture & Tech | Status / Help Wanted |
| :--- | :--- | :--- | :---: |
| **Milestone 9** | 🎮 **Gaming & Proton/Steam Layer** | Vulkan Implicit Layer (`VK_LAYER_OPENMEMGPU`), DXVK, Proton transparent texture spilling | `[Open for RFC]` 🚀 |
| **Milestone 10** | 🤖 **Local AI & LLM Inference** | `llama.cpp` integration, GGUF model layer splitting, KV cache auto-expansion | `[Open for RFC]` 🧠 |
| **Milestone 11** | 🪟 **Windows OS Support** | Direct3D 12 Reserved Resources, DXGI Memory Budgets, WDDM Virtual Memory, `windows-rs` | `[Help Wanted]` 💻 |
| **Milestone 12** | 🟢 **NVIDIA & CUDA Interoperability** | CUDA Driver API, `cuMemMap`, NVLink, Unified Memory (UVM), NVML telemetry | `[Help Wanted]` ⚡ |
| **Milestone 13** | 🔷 **Intel Arc & Xe Graphics** | Intel Level Zero, Mesa ANV Vulkan backend, Xe-LPG APU coordination | `[Research]` 🔬 |
| **Milestone 14** | 🌐 **CXL & Distributed Memory Pooling** | Compute Express Link (CXL 2.0/3.0), RDMA / RoCE cross-node GPU memory pooling | `[Long-Term]` 🔭 |

👉 **Read the full technical specification and contribution guides in [docs/roadmap/future_milestones.md](docs/roadmap/future_milestones.md).**

---

## 8. Building & Testing

### Prerequisites
- Linux with AMDGPU driver (or compatible Vulkan 1.3+ driver).
- Rust 1.80+ (`rustup default stable`).

```bash
# Run all workspace tests (unit, hardware, and integration)
cargo test --workspace

# Check formatting and clippy lints
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings

# Build all release binaries
cargo build --release
```

---

## 9. Contributing

We welcome pull requests, RFCs, and hardware benchmark submissions!
Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for architectural guidelines, coding style, and instructions on how to claim an upcoming milestone.

---

## 10. License

Dual-licensed under either:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE) or http://opensource.org/licenses/MIT)

at your option.
