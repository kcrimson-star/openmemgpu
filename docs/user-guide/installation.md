# Installation Guide: OpenMemGPU & VVRAM

**Document Version**: 1.0  

---

## 1. System Requirements

### Operating System
- **Linux**: Ubuntu 22.04 / 24.04 LTS, Pop!_OS 22.04 / 24.04, Fedora 39+, Arch Linux.
- **Windows**: Windows 11 (build 22621+) with AMD Adrenalin 24.x+ driver.

### Hardware
- **Primary Configuration**: Dual AMD GPU system (e.g. AMD Ryzen APU with Radeon Vega / RDNA2/3 iGPU + Radeon RX 5000/6000/7000 Series dGPU).
- **Secondary Configuration**: Single AMD dGPU or APU with host system RAM offloading.
- **Minimum Memory**: 16 GB system RAM recommended.

---

## 2. Toolchain Setup

### Step 1: Install Rust Toolchain
OpenMemGPU requires a standard Rust toolchain (version 1.80+):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

Verify the installation:
```bash
rustc --version
cargo --version
```

### Step 2: Ensure Vulkan Runtime and Loader
On Debian / Ubuntu / Pop!_OS:
```bash
sudo apt update
sudo apt install -y libvulkan1 vulkan-tools libvulkan-dev
```

Verify Vulkan GPU enumeration:
```bash
vulkaninfo --summary
```

---

## 3. Building OpenMemGPU

Clone or navigate to the workspace and build with Cargo:
```bash
cd openmemgpu
cargo build --release
```

Run test suite to verify the mock backends and unit tests:
```bash
cargo test --workspace
```
