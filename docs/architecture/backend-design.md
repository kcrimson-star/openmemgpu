# Backend Design & Abstraction Specification

**Document Version**: 1.0  
**Status**: Architecture Baseline  

---

## 1. Backend Architecture & Rust Traits

To isolate native graphics drivers, Vulkan interfaces, and platform-specific code, OpenMemGPU defines a set of backend traits in `openmemgpu-core`:

```rust
pub trait Backend: Send + Sync + 'static {
    type Device: Device;
    
    fn name(&self) -> &'static str;
    fn enumerate_devices(&self) -> Result<Vec<Self::Device>, CoreError>;
}

pub trait Device: Send + Sync + 'static {
    fn info(&self) -> &DeviceInfo;
    fn memory_domains(&self) -> &[MemoryDomainInfo];
    fn query_live_budget(&self) -> Result<Vec<DomainBudget>, CoreError>;
    
    fn allocate_buffer(
        &self, 
        domain_id: DomainId, 
        size_bytes: u64, 
        usage: BufferUsage
    ) -> Result<Box<dyn DeviceBuffer>, CoreError>;
    
    fn create_timeline_semaphore(&self, initial_value: u64) -> Result<Box<dyn TimelineSemaphore>, CoreError>;
    fn submit_compute(&self, command_batch: &CommandBatch) -> Result<(), CoreError>;
}

pub trait DeviceBuffer: Send + Sync {
    fn domain_id(&self) -> DomainId;
    fn size_bytes(&self) -> u64;
    fn device_address(&self) -> Option<u64>;
    fn map_host_ptr(&mut self) -> Result<*mut u8, CoreError>;
    fn unmap_host_ptr(&mut self);
}

pub trait TimelineSemaphore: Send + Sync {
    fn current_value(&self) -> Result<u64, CoreError>;
    fn wait_for_value(&self, value: u64, timeout_nanos: u64) -> Result<bool, CoreError>;
    fn signal_value(&self, value: u64) -> Result<(), CoreError>;
}
```

---

## 2. Concrete Backends

### 2.1 BackendVulkan (`crates/backend-vulkan`)
- Uses the `ash` crate (version 0.38+) to interface directly with Vulkan 1.4 loaders and Mesa RADV drivers.
- **Dynamic Function Loading**: Loads `libvulkan.so.1` at runtime; no dependency on C header files.
- **Queue Separation**: Explicitly exposes universal graphics/compute queue and dedicated asynchronous compute queues (`QUEUE_COMPUTE_BIT` without graphics).
- **Timeline Synchronization**: Implements `TimelineSemaphore` using `VkSemaphoreTypeKHR::TIMELINE`.
- **Memory Sub-allocation**: Integrates sub-allocation primitives to respect driver allocation limits.

### 2.2 BackendAmd (`crates/backend-amd`)
- Provides vendor-specific telemetry hooks on Linux systems.
- Reads `/sys/class/drm/card*/device/mem_info_*` to obtain authoritative driver-reported VRAM and GTT usage.
- Adjusts workgroup sizes based on GCN vs RDNA wavefront properties (Wave64 on Vega 6, Wave32 on Navi 10).

### 2.3 BackendMock (`crates/backend-mock` / test utils)
- Provides a software mock implementation of the `Backend` and `Device` traits.
- Enables deterministic CI tests of allocation logic, residency state machines, LRU eviction, and scheduler partitioning without requiring access to physical GPU hardware.
