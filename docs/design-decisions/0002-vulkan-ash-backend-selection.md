# ADR 0002: Vulkan 1.4 via `ash` as Primary GPU Backend

- **Status**: Accepted
- **Deciders**: OpenMemGPU Systems Architecture Team
- **Date**: October 21, 2026

## Context
The target hardware consists of an AMD Navi 10 discrete GPU and an AMD Renoir Vega 6 integrated GPU. Official ROCm packages do not stably support this heterogeneous mobile combination on standard Linux distributions. Both GPUs, however, fully support Vulkan 1.4 under the open-source Mesa RADV driver.

## Decision
Adopt Vulkan 1.4 as the primary cross-device compute and memory management backend, using the `ash` crate for Rust bindings.

## Consequences
- Full, vendor-supported access to both physical GPUs.
- Dynamic runtime loading of `libvulkan.so.1` without C header dependencies.
- Access to critical extensions: `VK_EXT_memory_budget`, `VK_KHR_external_memory_fd`, `VK_KHR_timeline_semaphore`.
- Higher-level abstractions like `wgpu` are rejected because they obscure memory types, custom heap selection, and external memory file descriptors.
