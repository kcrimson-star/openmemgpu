# Contributing to OpenMemGPU + VVRAM

Thank you for your interest in contributing to **OpenMemGPU** and the **VVRAM** systems research project!

---

## 1. Governing Code Principles

1. **All Implementation Code Must Be Rust**:
   - All core orchestration, memory management, scheduling, Vulkan backends, CLI tools, tests, and benchmarks must be implemented in safe, idiomatic Rust.
   - C/C++ or Python may not be introduced for project-owned components. External native vendor libraries (e.g. Vulkan loader `libvulkan.so.1`) are permitted via audited FFI bindings (such as `ash`).
2. **Minimize and Audit `unsafe`**:
   - `unsafe` blocks are restricted strictly to FFI boundary calls (Vulkan API invocations, pointer casting in buffer mapping).
   - Every `unsafe` block must be accompanied by a `// SAFETY:` comment explicitly proving preconditions and invariants.
3. **Research First, Measure Second, Implement Third, Validate Continuously**:
   - Never implement speculative memory tricks without verifying hardware support and measuring latency/bandwidth impacts.
   - Any architectural decision must be documented with rationale in `docs/design-decisions/`.

---

## 2. Development Workflow

### Toolchain Requirements
- Rust 1.80 or later (stable).
- `cargo`, `rustfmt`, `clippy`.
- Vulkan SDK / Loader 1.3+.

### Standard Commands
```bash
# Check formatting
cargo fmt --all -- --check

# Run linter
cargo clippy --workspace --all-targets -- -D warnings

# Run all unit tests
cargo test --workspace

# Run hardware discovery CLI
cargo run -p openmemgpu-cli -- discover
```

---

## 3. Claiming & Contributing to Future Milestones

We have outlined major upcoming milestones in [docs/roadmap/future_milestones.md](docs/roadmap/future_milestones.md):

- **Milestone 9**: Vulkan Layer for Transparent Game & App Expansion (`VK_LAYER_OPENMEMGPU`)
- **Milestone 10**: Native Local AI & LLM Inference Runtime (`llama.cpp` integration)
- **Milestone 11**: Windows OS Support (Direct3D 12, DXGI, WDDM Virtual Memory)
- **Milestone 12**: NVIDIA GPU Support & CUDA Interoperability (`cuMemMap`, NVLink)
- **Milestone 13**: Intel Arc & Xe Graphics Support
- **Milestone 14**: CXL & Distributed Networked Memory Pooling

### How to Claim a Milestone or Feature
1. Open an **RFC (Request for Comments)** issue titled `RFC: Milestone X - [Description]`.
2. Outline your target hardware, proposed crate architecture, and verification strategy.
3. Once aligned, submit PRs incrementally:
   - PR 1: Hardware discovery / adapter enumeration.
   - PR 2: Virtual memory allocation and buffer binding.
   - PR 3: Hardware benchmarks and integration tests.
