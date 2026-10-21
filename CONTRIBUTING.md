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
