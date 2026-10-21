use openmemgpu_core::device::DeviceId;
use openmemgpu_core::memory::{DomainId, MemoryDomainKind};
use openmemgpu_integrations::ai_model::ModelSlicer;
use openmemgpu_memory::ledger::PhysicalMemoryLedger;
use openmemgpu_scheduler::partitioner::AsymmetricPartitioner;
use openmemgpu_vvram::{PlacementPolicy, ResidencyState, VvramBufferDescriptor, VvramManager};
use std::sync::Arc;

#[test]
fn test_physical_memory_ledger_budget_enforcement() {
    // 6 GB VRAM, 14 GB Host RAM, 2 GB OS headroom -> 12 GB usable RAM
    let ledger = PhysicalMemoryLedger::new(
        6 * 1024 * 1024 * 1024,
        14 * 1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
    );

    let dgpu_vram = DomainId(0);
    let apu_gtt = DomainId(1);
    let dgpu_gtt = DomainId(2);

    // Discrete VRAM allocations
    assert!(ledger
        .reserve(dgpu_vram, MemoryDomainKind::DiscreteDedicatedVram, 4 * 1024 * 1024 * 1024)
        .is_ok());

    // Host-backed allocations (APU + dGPU GTT)
    assert!(ledger
        .reserve(apu_gtt, MemoryDomainKind::ApuSharedGtt, 7 * 1024 * 1024 * 1024)
        .is_ok());

    // Requesting another 6 GB on dGPU GTT exceeds the 12 GB usable RAM (7 + 6 = 13 > 12)
    let overflow = ledger.reserve(dgpu_gtt, MemoryDomainKind::DiscreteGtt, 6 * 1024 * 1024 * 1024);
    assert!(overflow.is_err());

    // Releasing 3 GB allows 4 GB allocation
    ledger.release(apu_gtt, MemoryDomainKind::ApuSharedGtt, 3 * 1024 * 1024 * 1024);
    assert!(ledger
        .reserve(dgpu_gtt, MemoryDomainKind::DiscreteGtt, 4 * 1024 * 1024 * 1024)
        .is_ok());
}

#[test]
fn test_asymmetric_partitioning_navi_and_renoir() {
    let dgpu = (DeviceId(0), 7.0); // Navi 10: 7.0 TFLOPS
    let igpu = (DeviceId(1), 1.1); // Renoir: 1.1 TFLOPS

    let (part_d, part_i) = AsymmetricPartitioner::partition(dgpu, igpu, 10_000);

    assert_eq!(part_d.slice_count + part_i.slice_count, 10_000);
    assert_eq!(part_d.slice_start, 0);
    assert_eq!(part_i.slice_start, part_d.slice_count);

    // Compute weight ratio ~86.4% to 13.6%
    assert!(part_d.compute_weight > 0.85 && part_d.compute_weight < 0.88);
    assert!(part_i.compute_weight > 0.12 && part_i.compute_weight < 0.15);
}

#[test]
fn test_ai_model_slicing_plan() {
    let dgpu_id = DeviceId(0);
    let igpu_id = DeviceId(1);

    // 32-layer transformer, 250 MB per layer (8 GB total model)
    // 5 GB dGPU budget -> 20 layers on dGPU, 12 layers on iGPU
    let plan = ModelSlicer::plan_placement(
        "Llama-3-8B-Q4",
        32,
        250 * 1024 * 1024,
        5 * 1024 * 1024 * 1024,
        dgpu_id,
        igpu_id,
    );

    assert_eq!(plan.layers.len(), 32);
    assert_eq!(plan.layers[0].target_device, dgpu_id);
    assert_eq!(plan.layers[19].target_device, dgpu_id);
    assert_eq!(plan.layers[20].target_device, igpu_id);
    assert_eq!(plan.layers[31].target_device, igpu_id);
}

#[test]
fn test_vvram_residency_lifecycle() {
    let ledger = Arc::new(PhysicalMemoryLedger::new(
        6 * 1024 * 1024 * 1024,
        14 * 1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
    ));

    let vvram = VvramManager::new(
        ledger.clone(),
        DomainId(0),
        DomainId(1),
        DomainId(2),
    );

    let buf = vvram
        .allocate(VvramBufferDescriptor {
            name: "kv_cache_tensor".into(),
            size_bytes: 512 * 1024 * 1024,
            policy: PlacementPolicy::PreferDiscreteVram,
        })
        .unwrap();

    assert_eq!(buf.residency, ResidencyState::AllocatedLogical);

    vvram.materialize(buf.id).unwrap();
    assert_eq!(ledger.dedicated_vram_allocated(), 512 * 1024 * 1024);

    vvram.free(buf.id).unwrap();
    assert_eq!(ledger.dedicated_vram_allocated(), 0);
}
