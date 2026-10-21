# ADR 0005: Pure-Rust Shader Compilation via Naga & Wavefront Size Handling

- **Status**: Accepted
- **Deciders**: OpenMemGPU Systems Architecture Team
- **Date**: October 21, 2026

## Context
Phase 3 requires a maintained shader compilation workflow compatible with the Rust-only project-owned code requirement. External toolchains like `glslangValidator` or `glslc` are not installed by default on host systems and introduce native C++ build dependencies.
Furthermore, physical hardware testing revealed that Mesa RADV defaults to `subgroup_size: 64` (Wave64) on both Navi 10 and Renoir APUs unless specialized extensions are enabled.

## Decision
1. Adopt `naga` (v24.0.0, developed by gfx-rs/wgpu) as the pure-Rust compute shader compiler, translating WGSL shader representations directly into valid SPIR-V 1.3 bytecode at runtime.
2. Parameterize compute shader workgroup sizes by the target device's reported `VkPhysicalDeviceSubgroupProperties.subgroup_size` (e.g. 64 on both GPUs under RADV).

## Consequences
- Eliminates all external compiler binaries and C/C++ dependencies.
- Zero-overhead runtime compilation (< 1 ms per shader).
- Full compliance with the project's strict pure-Rust mandate.
