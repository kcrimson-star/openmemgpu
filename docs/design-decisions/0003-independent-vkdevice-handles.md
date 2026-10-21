# ADR 0003: Independent `VkDevice` Instances for Heterogeneous GPUs

- **Status**: Accepted
- **Deciders**: OpenMemGPU Systems Architecture Team
- **Date**: October 21, 2026

## Context
Vulkan device groups (`VK_KHR_device_group`) allow multiple physical GPUs to be managed under a single logical `VkDevice`, but hardware drivers restrict device groups to homogeneous GPUs (e.g. identical dual Navi 10 cards). The driver enumerates our Navi 10 dGPU and Renoir iGPU into separate device groups of count 1.

## Decision
OpenMemGPU will initialize and manage separate, independent `VkDevice` instances for the discrete and integrated GPUs, coordinating memory sharing via host staging buffers and `VK_KHR_external_memory_fd`.

## Consequences
- Clean separation of device lifecycles, error states, and queues.
- Asymmetric wavefront and compute pipeline compilation per device.
- Software-level coordination of synchronization via timeline semaphores.
