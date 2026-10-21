# Usage Guide: OpenMemGPU & VVRAM CLI

---

## 1. CLI Commands Overview

The `openmemgpu-cli` binary provides management, diagnostics, and benchmarking tools:

```bash
# Display general help
openmemgpu-cli --help

# Discover hardware and inspect Vulkan memory heaps
openmemgpu-cli discover

# Query live VVRAM status and driver memory budgets
openmemgpu-cli status

# Run allocation benchmark to verify allocatable capacity
openmemgpu-cli bench --mode alloc

# Run transfer bandwidth benchmark (H2D, D2H, D2D)
openmemgpu-cli bench --mode transfer

# Run compute test across single and multi-device configurations
openmemgpu-cli bench --mode compute
```

---

## 2. Using the Rust API

Compatible AI runtimes integrate with VVRAM via the `openmemgpu_vvram` crate:

```rust
use openmemgpu_core::error::CoreError;
use openmemgpu_vvram::{VvramManager, VvramBufferDescriptor, PlacementPolicy};

fn main() -> Result<(), CoreError> {
    // 1. Initialize VVRAM Manager
    let mut vvram = VvramManager::init_default()?;

    // 2. Query available capacity across all physical domains
    let report = vvram.capacity_report();
    println!("Dedicated VRAM: {} MB", report.dedicated_vram_total / 1024 / 1024);
    println!("Host Available: {} MB", report.host_ram_available / 1024 / 1024);

    // 3. Allocate a virtual buffer (e.g., 2 GB model layer tensor)
    let tensor = vvram.allocate(VvramBufferDescriptor {
        name: "transformer_layer_0_qkv".into(),
        size_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
        policy: PlacementPolicy::PreferDiscreteVram,
    })?;

    // 4. Materialize onto physical device
    vvram.materialize(&tensor)?;

    Ok(())
}
```
