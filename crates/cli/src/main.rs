use clap::{Parser, Subcommand};
use openmemgpu_backend_amd::AmdBackend;
use openmemgpu_backend_vulkan::context::VulkanContext;
use openmemgpu_backend_vulkan::device::VulkanDevice;
use openmemgpu_backend_vulkan::physical_device::PhysicalDeviceEnumerator;
use openmemgpu_bench::gemm::GemmBenchmarkRunner;
use openmemgpu_bench::multi_device::MultiDeviceWorkloadRunner;
use openmemgpu_bench::runner::BenchmarkRunner;
use openmemgpu_device_discovery::DeviceDiscovery;
use openmemgpu_memory::ledger::PhysicalMemoryLedger;
use openmemgpu_scheduler::WorkloadScheduler;
use openmemgpu_vvram::{DynamicVramExpander, PlacementPolicy, VvramBufferDescriptor, VvramManager};
use std::fs;
use std::path::Path;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "openmemgpu-cli")]
#[command(about = "OpenMemGPU & VVRAM Management, Diagnostics, and Benchmarking CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Discover GPU devices and inspect memory domains
    Discover,
    /// Query live VVRAM status, kernel DRM sysfs metrics, and Vulkan driver budgets
    Status,
    /// Execute a sample VVRAM allocation and residency demonstration
    Demo,
    /// Run physical GPU compute benchmarks with reproducible JSON output
    Bench {
        /// Number of elements in vector multiplication (default: 1,048,576 = 1M elements, 4MB per buffer)
        #[arg(short, long, default_value_t = 1_048_576)]
        elements: usize,

        /// Number of warm-up iterations
        #[arg(short, long, default_value_t = 5)]
        warmup: usize,

        /// Number of measured iterations
        #[arg(short, long, default_value_t = 20)]
        runs: usize,

        /// Output path for machine-readable JSON report
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Benchmark dedicated VRAM vs host-visible memory tiers on dGPU and iGPU
    BenchTiering {
        /// Number of warm-up iterations
        #[arg(short, long, default_value_t = 3)]
        warmup: usize,

        /// Number of measured iterations
        #[arg(short, long, default_value_t = 10)]
        runs: usize,

        /// Output path for machine-readable JSON report
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Benchmark bidirectional cross-GPU data transfers (dGPU <-> iGPU) with DMA-BUF probe
    BenchTransfer {
        /// Number of warm-up iterations
        #[arg(short, long, default_value_t = 3)]
        warmup: usize,

        /// Number of measured iterations
        #[arg(short, long, default_value_t = 10)]
        runs: usize,

        /// Output path for machine-readable JSON report
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Benchmark multi-device execution (dGPU-only vs iGPU-only vs Partitioned Multi-GPU)
    BenchMultidevice {
        /// Buffer sizes in elements to benchmark (default: 16384, 1048576, 4194304, 16777216)
        #[arg(short, long, value_delimiter = ',')]
        elements: Option<Vec<usize>>,

        /// Number of warm-up iterations
        #[arg(short, long, default_value_t = 3)]
        warmup: usize,

        /// Number of measured iterations
        #[arg(short, long, default_value_t = 10)]
        runs: usize,

        /// Output path for machine-readable JSON report
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Evaluate scheduler partitioning and execution mode decisions for a given workload size
    Schedule {
        /// Workload size in elements (f32)
        #[arg(short, long, default_value_t = 1_048_576)]
        elements: usize,
    },
    /// Benchmark high-arithmetic-intensity Matrix Multiplication (GEMM: C = A * B)
    Gemm {
        /// Matrix dimension N (for N x N matrix, default: 512)
        #[arg(short, long, default_value_t = 512)]
        size: u32,

        /// Number of measured iterations
        #[arg(short, long, default_value_t = 5)]
        runs: usize,

        /// Output path for machine-readable JSON report
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Query or set dynamic VRAM capacity expansion (borrowing system RAM via iGPU)
    Expand {
        /// Additional VRAM in gigabytes to borrow from iGPU (e.g. 2, 4, 6)
        #[arg(short, long)]
        add_gb: Option<u32>,
    },
    /// Probe Linux DRM / Vulkan DMA-BUF export and import capabilities
    ProbeDmaBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Discover => {
            println!("=== OpenMemGPU Hardware Discovery ===");
            let devices = DeviceDiscovery::discover_from_sysfs()?;
            for dev in &devices {
                println!("\nDevice {}: {}", dev.id.0, dev.name);
                println!("  Architecture: {:?}", dev.architecture);
                println!("  Type: {:?}", dev.kind);
                println!("  PCI Bus: {:?}", dev.capabilities.pci_bus_id);
                println!(
                    "  Native Wavefront: {} threads",
                    dev.capabilities.native_wavefront_size
                );
                println!(
                    "  Estimated FP32: {:.1} TFLOPS",
                    dev.capabilities.estimated_fp32_tflops
                );
            }

            // Also inspect Vulkan devices
            println!("\n=== Vulkan 1.4 Physical Device Enumeration ===");
            if let Ok(ctx) = VulkanContext::new() {
                if let Ok(vk_devices) = PhysicalDeviceEnumerator::enumerate(&ctx.instance) {
                    for (_, d) in vk_devices {
                        println!("\nVulkan Device: {}", d.name);
                        println!("  Device Type: {:?}", d.kind);
                        println!("  API Version: {}", d.api_version);
                        println!("  Driver Version: {}", d.driver_version);
                        println!(
                            "  PCI Address: {:?}",
                            d.pci_bus.as_ref().map(|p| &p.pci_address)
                        );
                        println!("  DRM Render Node: {:?}", d.drm_render_node);
                        println!("  Subgroup Size: {} threads", d.subgroup_size);
                        println!("  Heaps:");
                        for h in &d.heaps {
                            println!(
                                "    Heap {}: size={:>5} MB, dev_local={:<5}, budget={:?} MB, usage={:?} MB",
                                h.index,
                                h.size_bytes / 1024 / 1024,
                                h.is_device_local,
                                h.driver_budget_bytes.map(|b| b / 1024 / 1024),
                                h.driver_usage_bytes.map(|u| u / 1024 / 1024)
                            );
                        }
                    }
                }
            }
        }
        Commands::Status => {
            println!("=== OpenMemGPU & AMD DRM Telemetry Status ===");
            for card in ["card1", "card2"] {
                if let Ok(metrics) = AmdBackend::read_card_metrics(card) {
                    println!("\n[{}] Kernel DRM Sysfs Metrics:", card);
                    println!(
                        "  VRAM Total: {} MB | Used: {} MB",
                        metrics.vram_total_bytes / 1024 / 1024,
                        metrics.vram_used_bytes / 1024 / 1024
                    );
                    println!(
                        "  GTT Total:  {} MB | Used: {} MB",
                        metrics.gtt_total_bytes / 1024 / 1024,
                        metrics.gtt_used_bytes / 1024 / 1024
                    );
                    println!(
                        "  Vis VRAM:   {} MB (Resizable BAR)",
                        metrics.vis_vram_total_bytes / 1024 / 1024
                    );
                }
            }

            println!("\n=== Vulkan Live Memory Budget (VK_EXT_memory_budget) ===");
            if let Ok(ctx) = VulkanContext::new() {
                if let Ok(vk_devices) = PhysicalDeviceEnumerator::enumerate(&ctx.instance) {
                    for (_, d) in vk_devices {
                        if d.is_discrete() || d.is_integrated() {
                            println!("\n[{}] Driver Memory Heaps:", d.name);
                            for h in &d.heaps {
                                let label = if h.is_device_local {
                                    "Device-Local (VRAM)"
                                } else {
                                    "Host-Visible (GTT / Host RAM)"
                                };
                                println!(
                                    "  Heap {} [{}]: Size={:>5} MB, Budget={:>5} MB, Usage={:>4} MB",
                                    h.index,
                                    label,
                                    h.size_bytes / 1024 / 1024,
                                    h.driver_budget_bytes.unwrap_or(0) / 1024 / 1024,
                                    h.driver_usage_bytes.unwrap_or(0) / 1024 / 1024
                                );
                            }
                        }
                    }
                }
            }
        }
        Commands::Demo => {
            println!("=== VVRAM Allocation & Physical Ledger Demonstration ===");
            let ledger = Arc::new(PhysicalMemoryLedger::new(
                6 * 1024 * 1024 * 1024,
                14 * 1024 * 1024 * 1024,
                2 * 1024 * 1024 * 1024,
            ));

            let vvram = VvramManager::new(
                ledger.clone(),
                openmemgpu_core::memory::DomainId(0),
                openmemgpu_core::memory::DomainId(1),
                openmemgpu_core::memory::DomainId(2),
            );

            println!("Initial Capacity Report:");
            let r1 = vvram.capacity_report();
            println!(
                "  Dedicated VRAM: {} MB",
                r1.dedicated_vram_total_bytes / 1024 / 1024
            );
            println!(
                "  Host RAM Available: {} MB",
                r1.host_ram_available_bytes / 1024 / 1024
            );

            println!("\nAllocating 2 GB Virtual Buffer (transformer_layer_0)...");
            let buf = vvram.allocate(VvramBufferDescriptor {
                name: "transformer_layer_0".into(),
                size_bytes: 2 * 1024 * 1024 * 1024,
                policy: PlacementPolicy::PreferDiscreteVram,
            })?;
            println!("  Allocated Logical Buffer ID: {:?}", buf.id);

            println!("Materializing into Discrete VRAM (Logical)...");
            vvram.materialize_logical(buf.id)?;
            let r2 = vvram.capacity_report();
            println!(
                "  Dedicated VRAM Allocated: {} MB",
                r2.dedicated_vram_allocated_bytes / 1024 / 1024
            );
            println!(
                "  Dedicated VRAM Remaining: {} MB",
                r2.dedicated_vram_available_bytes / 1024 / 1024
            );

            println!("\nReleasing Buffer...");
            vvram.free(buf.id)?;
            let r3 = vvram.capacity_report();
            println!(
                "  Dedicated VRAM Allocated: {} MB",
                r3.dedicated_vram_allocated_bytes / 1024 / 1024
            );
            println!("Demonstration complete.");
        }
        Commands::Bench {
            elements,
            warmup,
            runs,
            output,
        } => {
            println!("=== OpenMemGPU Physical GPU Compute Benchmark Suite ===");
            println!("Workload: Element-wise Vector Multiply (C[i] = A[i] * B[i])");
            println!(
                "Configuration: {} elements ({} KB per buffer)",
                elements,
                (elements * 4) / 1024
            );
            println!(
                "Methodology: {} warmup iterations, {} measured iterations\n",
                warmup, runs
            );

            let runner = BenchmarkRunner::new();
            let report = runner.run_compute_suite(elements, warmup, runs)?;

            for dev in &report.device_reports {
                println!("--------------------------------------------------");
                println!("Device:      {}", dev.device_name);
                println!(
                    "Type:        {} | Wavefront: {} threads",
                    dev.device_kind, dev.subgroup_size
                );
                println!("PCI Bus:     {:?}", dev.pci_address);
                println!("Driver:      {}", dev.driver_version);
                println!("Evidence:    {:?}", dev.evidence);
                println!(
                    "Correctness: {}",
                    if dev.correctness_verified {
                        "VERIFIED (100% MATCH)"
                    } else {
                        "FAILED"
                    }
                );
                println!(
                    "VRAM Budget: {:?} MB (Usage: {:?} MB)",
                    dev.vram_budget_bytes.map(|b| b / 1024 / 1024),
                    dev.vram_usage_bytes.map(|u| u / 1024 / 1024)
                );
                println!("Performance Metrics (Steady-State Median):");
                println!(
                    "  Pipeline Init (one-time): {:.1} us",
                    dev.init_pipeline_time_us
                );
                println!(
                    "  Buffer Allocation:        {:.1} us",
                    dev.alloc_time_us_median
                );
                println!(
                    "  Host-to-Device Upload:    {:.1} us",
                    dev.upload_time_us_median
                );
                println!(
                    "  Compute Dispatch:         {:.1} us",
                    dev.dispatch_time_us_median
                );
                println!(
                    "  Device-to-Host Download:  {:.1} us",
                    dev.download_time_us_median
                );
                println!(
                    "  Total Steady-State:       {:.1} us",
                    dev.total_steady_state_us_median
                );
                println!(
                    "  Effective Bandwidth:      {:.2} GB/s",
                    dev.effective_bandwidth_gbps
                );
            }
            println!("--------------------------------------------------");

            if let Some(ref path_str) = output {
                let out_path = Path::new(path_str);
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let json_data = serde_json::to_string_pretty(&report)?;
                fs::write(out_path, json_data)?;
                println!("\n[SUCCESS] Benchmark report saved to: {}", path_str);
            }
        }
        Commands::BenchTiering {
            warmup,
            runs,
            output,
        } => {
            println!("=== OpenMemGPU Memory Tiering Benchmark Suite ===");
            println!("Comparing Dedicated Device-Local VRAM (with Staging) vs Host-Visible GTT");
            println!(
                "Buffer Sizes: 64 KB (16,384 f32), 4 MB (1,048,576 f32), 16 MB (4,194,304 f32)\n"
            );

            let runner = BenchmarkRunner::new();
            let sizes = [16_384usize, 1_048_576, 4_194_304];
            let report = runner.run_memory_tiering_suite(&sizes, warmup, runs)?;

            for case in &report.cases {
                println!("--------------------------------------------------------------------------------");
                println!("Device:      {}", case.device_name);
                println!("Memory Tier: {}", case.memory_tier);
                println!(
                    "Buffer Size: {} KB ({} bytes)",
                    case.buffer_size_bytes / 1024,
                    case.buffer_size_bytes
                );
                println!(
                    "Correctness: {}",
                    if case.correctness_verified {
                        "VERIFIED (100% MATCH)"
                    } else {
                        "FAILED"
                    }
                );
                println!("Timing Breakdown (Steady-State Median):");
                println!("  Allocation:              {:>8.1} us", case.alloc_time_us);
                println!(
                    "  Upload (Staged/Direct):  {:>8.1} us",
                    case.upload_time_us_median
                );
                println!(
                    "  Compute Kernel Dispatch: {:>8.1} us",
                    case.dispatch_time_us_median
                );
                println!(
                    "  Download (Staged/Direct):{:>8.1} us",
                    case.download_time_us_median
                );
                println!(
                    "  End-to-End Total:        {:>8.1} us",
                    case.total_steady_state_us_median
                );
                println!("Throughput:");
                println!(
                    "  Compute Kernel Bandwidth: {:>7.2} GB/s",
                    case.compute_kernel_bandwidth_gbps
                );
                println!(
                    "  End-to-End Throughput:    {:>7.2} GB/s",
                    case.end_to_end_bandwidth_gbps
                );
            }
            println!(
                "--------------------------------------------------------------------------------"
            );

            if let Some(ref path_str) = output {
                let out_path = Path::new(path_str);
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let json_data = serde_json::to_string_pretty(&report)?;
                fs::write(out_path, json_data)?;
                println!("\n[SUCCESS] Memory tiering report saved to: {}", path_str);
            }
        }
        Commands::BenchTransfer {
            warmup,
            runs,
            output,
        } => {
            println!("=== OpenMemGPU Cross-GPU Data Transfer Benchmark Suite ===");
            println!("Bidirectional Host-Mediated Staging Transfers (dGPU <-> iGPU)");
            println!(
                "Buffer Sizes: 64 KB (16,384 f32), 4 MB (1,048,576 f32), 16 MB (4,194,304 f32)\n"
            );

            let runner = BenchmarkRunner::new();
            let sizes = [16_384usize, 1_048_576, 4_194_304];
            let report = runner.run_cross_gpu_suite(&sizes, warmup, runs)?;

            for case in &report.cases {
                println!("--------------------------------------------------------------------------------");
                println!("Transfer Direction:  {}", case.direction);
                println!(
                    "Buffer Size:         {} KB ({} bytes)",
                    case.buffer_size_bytes / 1024,
                    case.buffer_size_bytes
                );
                println!(
                    "Integrity:           {}",
                    if case.correctness_verified {
                        "VERIFIED (100% BYTE-FOR-BYTE)"
                    } else {
                        "FAILED"
                    }
                );
                println!("Timing Breakdown (Steady-State Median):");
                println!(
                    "  Source Download:    {:>8.1} us",
                    case.src_download_us_median
                );
                println!(
                    "  Destination Upload: {:>8.1} us",
                    case.dst_upload_us_median
                );
                println!(
                    "  Total Transfer:     {:>8.1} us",
                    case.total_transfer_us_median
                );
                println!(
                    "Effective Throughput: {:>7.2} GB/s",
                    case.effective_bandwidth_gbps
                );
            }
            println!(
                "--------------------------------------------------------------------------------"
            );

            if let Some(ref dma_probe) = report.dma_buf_feasibility {
                println!("\n=== Linux DRM / Vulkan DMA-BUF Feasibility Summary ===");
                println!(
                    "Direct Cross-GPU DMA-BUF Supported: {}",
                    dma_probe.direct_cross_gpu_dma_supported
                );
                println!("Architectural Summary: {}", dma_probe.architectural_summary);
                for d in &dma_probe.devices {
                    println!(
                        "  [{}]: Exportable={}, Importable={}, Compatible Handles={:#x}",
                        d.device_name,
                        d.dma_buf_exportable,
                        d.dma_buf_importable,
                        d.compatible_handle_types
                    );
                }
            }

            if let Some(ref path_str) = output {
                let out_path = Path::new(path_str);
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let json_data = serde_json::to_string_pretty(&report)?;
                fs::write(out_path, json_data)?;
                println!(
                    "\n[SUCCESS] Cross-GPU transfer report saved to: {}",
                    path_str
                );
            }
        }
        Commands::BenchMultidevice {
            elements,
            warmup,
            runs,
            output,
        } => {
            println!("=== OpenMemGPU Multi-Device Execution Benchmark Suite ===");
            println!(
                "Workload: Two-Stage Compute (Stage 1: Z = A * B, Stage 2: Out = Z * 1.5 + 0.5)"
            );
            println!("Hardware: dGPU (AMD Radeon RX 5600M) + iGPU (AMD Radeon Graphics Vega 6)");
            println!(
                "Methodology: {} warmup iterations, {} measured iterations\n",
                warmup, runs
            );

            let sizes = elements.unwrap_or_else(|| vec![16_384, 1_048_576, 4_194_304, 16_777_216]);
            let runner = MultiDeviceWorkloadRunner::new()?;
            let report = runner.run_suite(&sizes, warmup, runs)?;

            for comp in &report.comparisons {
                println!("================================================================================");
                println!(
                    "Buffer Size: {} elements ({} KB, {:.2} MB per buffer)",
                    comp.total_elements,
                    comp.buffer_size_bytes / 1024,
                    comp.buffer_size_bytes as f64 / (1024.0 * 1024.0)
                );
                println!("Outcome:     {}", comp.outcome_summary);
                println!(
                    "Speedup vs dGPU: {:.3}x | Speedup vs iGPU: {:.3}x",
                    comp.speedup_vs_dgpu, comp.speedup_vs_igpu
                );
                println!("--------------------------------------------------------------------------------");
                println!(
                    "{:<24} | {:>12} | {:>12} | {:>12} | {:>10}",
                    "Execution Mode", "Wallclock Med", "Wallclock p95", "Throughput", "Verified"
                );
                println!(
                    "{:<24}-+-{:>12}-+-{:>12}-+-{:>12}-+-{:>10}",
                    "-".repeat(24),
                    "-".repeat(12),
                    "-".repeat(12),
                    "-".repeat(12),
                    "-".repeat(10)
                );
                for res in [&comp.dgpu_only, &comp.igpu_only, &comp.partitioned_multigpu] {
                    let mode_str = match res.mode {
                        openmemgpu_scheduler::WorkloadExecutionMode::DiscreteGpuOnly => {
                            "1. Discrete GPU Only"
                        }
                        openmemgpu_scheduler::WorkloadExecutionMode::IntegratedGpuOnly => {
                            "2. Integrated GPU Only"
                        }
                        openmemgpu_scheduler::WorkloadExecutionMode::PartitionedMultiGpu => {
                            "3. Partitioned Multi-GPU"
                        }
                    };
                    println!(
                        "{:<24} | {:>9.1} us | {:>9.1} us | {:>8.2} GB/s | {:>10}",
                        mode_str,
                        res.end_to_end_wallclock_us.median,
                        res.end_to_end_wallclock_us.p95,
                        res.effective_throughput_gbps,
                        if res.correctness_verified {
                            "100% MATCH"
                        } else {
                            "FAILED"
                        }
                    );
                }
                println!("--------------------------------------------------------------------------------");
                println!("Partitioned Multi-GPU Detailed Latency Breakdown (Median):");
                let part = &comp.partitioned_multigpu;
                println!(
                    "  dGPU Elements:           {:>10} ({:.1}%)",
                    part.dgpu_elements,
                    (part.dgpu_elements as f64 / comp.total_elements as f64) * 100.0
                );
                println!(
                    "  iGPU Elements:           {:>10} ({:.1}%)",
                    part.igpu_elements,
                    (part.igpu_elements as f64 / comp.total_elements as f64) * 100.0
                );
                println!(
                    "  Buffer Allocation:       {:>10.1} us",
                    part.alloc_time_us.median
                );
                println!(
                    "  Input Upload:            {:>10.1} us",
                    part.upload_time_us.median
                );
                println!(
                    "  Parallel Compute Phase:  {:>10.1} us (dGPU: {:.1} us, iGPU: {:.1} us)",
                    part.parallel_compute_time_us.median,
                    part.dgpu_compute_time_us.median,
                    part.igpu_compute_time_us.median
                );
                println!(
                    "  Cross-Device Transfer:   {:>10.1} us",
                    part.cross_device_transfer_time_us.median
                );
                println!(
                    "  Output Download:         {:>10.1} us",
                    part.download_time_us.median
                );
                println!(
                    "  Sync & Merge:            {:>10.1} us",
                    part.sync_and_merge_time_us.median
                );
                println!(
                    "  Total End-to-End:        {:>10.1} us",
                    part.end_to_end_wallclock_us.median
                );
            }
            println!(
                "================================================================================"
            );

            if let Some(ref path_str) = output {
                let out_path = Path::new(path_str);
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let json_data = serde_json::to_string_pretty(&report)?;
                fs::write(out_path, json_data)?;
                println!(
                    "\n[SUCCESS] Multi-device benchmark report saved to: {}",
                    path_str
                );
            }
        }
        Commands::Schedule { elements } => {
            println!("=== OpenMemGPU Workload Scheduling Decision ===");
            let scheduler = WorkloadScheduler::default();
            let decision = scheduler.decide(elements);
            println!(
                "Workload Elements:   {} ({} KB, {:.2} MB)",
                elements,
                (elements * 4) / 1024,
                (elements * 4) as f64 / (1024.0 * 1024.0)
            );
            println!("Selected Mode:       {:?}", decision.selected_mode);
            println!("Decision Rationale:  {}", decision.rationale);
            println!(
                "Minimum Threshold:   {} elements ({} KB)",
                scheduler.min_partition_elements,
                (scheduler.min_partition_elements * 4) / 1024
            );
            println!("dGPU TFLOPS:         {:.1}", scheduler.dgpu_tflops);
            println!("iGPU TFLOPS:         {:.1}", scheduler.igpu_tflops);
            println!("Estimated Latencies:");
            println!(
                "  dGPU Only:         {:>8.1} us",
                decision.estimated_dgpu_time_us
            );
            println!(
                "  iGPU Only:         {:>8.1} us",
                decision.estimated_igpu_time_us
            );
            println!(
                "  Transfer Overhead: {:>8.1} us",
                decision.estimated_transfer_overhead_us
            );
            println!(
                "  Partitioned Total: {:>8.1} us",
                decision.estimated_partitioned_time_us
            );
            if decision.selected_mode
                == openmemgpu_scheduler::WorkloadExecutionMode::PartitionedMultiGpu
            {
                println!("\nPartition Breakdown:");
                println!(
                    "  dGPU (Device 0): {} elements ({:.1}%)",
                    decision.dgpu_slice_count,
                    (decision.dgpu_slice_count as f64 / elements as f64) * 100.0
                );
                println!(
                    "  iGPU (Device 1): {} elements ({:.1}%)",
                    decision.igpu_slice_count,
                    (decision.igpu_slice_count as f64 / elements as f64) * 100.0
                );
            }
        }
        Commands::Gemm { size, runs, output } => {
            println!("=== OpenMemGPU High-Arithmetic-Intensity Matrix Multiplication (GEMM) ===");
            println!(
                "Workload: C[{}x{}] = A[{}x{}] * B[{}x{}]",
                size, size, size, size, size, size
            );
            let total_ops = 2.0 * (size as f64).powi(3);
            println!(
                "Total Compute: {:.3} GFLOPs ({:.0} operations)",
                total_ops / 1.0e9,
                total_ops
            );
            println!(
                "Arithmetic Intensity: {:.2} FLOP/byte\n",
                (size as f64) / 6.0
            );

            let runner = GemmBenchmarkRunner::new()?;
            let comp = runner.run_comparison(size, size, size, runs)?;

            println!(
                "--------------------------------------------------------------------------------"
            );
            println!(
                "{:<24} | {:>12} | {:>12} | {:>10}",
                "Execution Mode", "Duration", "Throughput", "Verified"
            );
            println!(
                "{:<24}-+-{:>12}-+-{:>12}-+-{:>10}",
                "-".repeat(24),
                "-".repeat(12),
                "-".repeat(12),
                "-".repeat(10)
            );
            println!(
                "{:<24} | {:>9.1} us | {:>9.2} GFLOPS | {:>10}",
                "1. Discrete GPU Only",
                comp.dgpu_only.wallclock_time_us,
                comp.dgpu_only.gflops,
                if comp.dgpu_only.correctness_verified {
                    "100% MATCH"
                } else {
                    "FAILED"
                }
            );
            println!(
                "{:<24} | {:>9.1} us | {:>9.2} GFLOPS | {:>10}",
                "2. Integrated GPU Only",
                comp.igpu_only.wallclock_time_us,
                comp.igpu_only.gflops,
                if comp.igpu_only.correctness_verified {
                    "100% MATCH"
                } else {
                    "FAILED"
                }
            );
            println!(
                "{:<24} | {:>9.1} us | {:>9.2} GFLOPS | {:>10}",
                "3. Partitioned Multi-GPU",
                comp.partitioned_multigpu.wallclock_time_us,
                comp.partitioned_multigpu.gflops,
                if comp.partitioned_multigpu.correctness_verified {
                    "100% MATCH"
                } else {
                    "FAILED"
                }
            );
            println!(
                "--------------------------------------------------------------------------------"
            );
            println!(
                "Speedup vs dGPU: {:.3}x | Speedup vs iGPU: {:.3}x",
                comp.speedup_vs_dgpu, comp.speedup_vs_igpu
            );
            println!("Breakdown of Partitioned Multi-GPU:");
            println!(
                "  Concurrent Compute: {:>10.1} us",
                comp.partitioned_multigpu.compute_time_us
            );
            println!(
                "  Cross-Device Xfer:  {:>10.1} us",
                comp.partitioned_multigpu.transfer_time_us
            );
            println!(
                "  Total Wallclock:    {:>10.1} us",
                comp.partitioned_multigpu.wallclock_time_us
            );

            if let Some(ref path_str) = output {
                let out_path = Path::new(path_str);
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let json_data = serde_json::to_string_pretty(&comp)?;
                fs::write(out_path, json_data)?;
                println!("\n[SUCCESS] GEMM benchmark report saved to: {}", path_str);
            }
        }
        Commands::Expand { add_gb } => {
            println!("=== OpenMemGPU Dynamic VRAM Capacity Expander ===");
            let ctx = Arc::new(VulkanContext::new()?);
            let enumerated = PhysicalDeviceEnumerator::enumerate(&ctx.instance)?;
            let dgpu_entry = enumerated
                .iter()
                .find(|(_, d)| d.kind == openmemgpu_core::device::DeviceKind::DiscreteGpu);
            let igpu_entry = enumerated
                .iter()
                .find(|(_, d)| d.kind == openmemgpu_core::device::DeviceKind::IntegratedGpu);

            let dgpu = dgpu_entry
                .map(|&(p, ref d)| VulkanDevice::new_with_context(ctx.clone(), p, d.clone()).unwrap());
            let igpu = igpu_entry
                .map(|&(p, ref d)| VulkanDevice::new_with_context(ctx.clone(), p, d.clone()).unwrap());

            let ledger = Arc::new(PhysicalMemoryLedger::new(
                6 * 1024 * 1024 * 1024,
                14 * 1024 * 1024 * 1024,
                2 * 1024 * 1024 * 1024,
            ));
            let manager = Arc::new(VvramManager::new(
                ledger,
                openmemgpu_core::memory::DomainId(0),
                openmemgpu_core::memory::DomainId(1),
                openmemgpu_core::memory::DomainId(2),
            ));

            let expander = DynamicVramExpander::new(manager, dgpu, igpu, Some(ctx.clone()))?;
            let mem = &expander.memory_info;

            println!("System Hardware Discovery:");
            println!(
                "  Host System RAM:       {:>6.1} GB total ({:.1} GB available)",
                mem.total_ram_bytes as f64 / 1e9,
                mem.available_ram_bytes as f64 / 1e9
            );
            println!(
                "  OS & Desktop Reserve:  {:>6.1} GB (protected)",
                mem.os_reserved_ram_bytes as f64 / 1e9
            );
            println!(
                "  dGPU Dedicated VRAM:   {:>6.1} GB (Navi 10 GDDR6)",
                mem.dgpu_vram_bytes as f64 / 1e9
            );
            println!(
                "  Max Safe Expansion:    {:>6.1} GB (via Renoir APU)",
                mem.max_expandable_bytes as f64 / 1e9
            );
            println!("  Supported Step Steps:  {:?} GB", mem.recommended_steps_gb);

            if let Some(add) = add_gb {
                println!("\nApplying Dynamic VRAM Expansion (+{} GB)...", add);
                let status = expander.set_expansion(add)?;
                println!("  [SUCCESS] Unified VVRAM Provisioned:");
                println!("    Dedicated dGPU:  {:.1} GB", status.base_dgpu_vram_gb);
                println!(
                    "    Expanded iGPU:   +{:.1} GB",
                    status.expanded_igpu_vram_gb
                );
                println!("    -----------------------------");
                println!(
                    "    TOTAL VVRAM:     {:.1} GB Unified Capacity",
                    status.total_vvram_gb
                );
                println!(
                    "    Backing VRAM:    {} MB active backing allocation",
                    status.backing_allocated_bytes / 1024 / 1024
                );
            } else {
                let status = expander.status();
                println!("\nCurrent VVRAM Status:");
                println!("  Dedicated dGPU:    {:.1} GB", status.base_dgpu_vram_gb);
                println!(
                    "  Expanded iGPU:     +{:.1} GB",
                    status.expanded_igpu_vram_gb
                );
                println!("  Total VVRAM Pool:  {:.1} GB", status.total_vvram_gb);
                println!("\nUse `openmemgpu-cli expand --add-gb <N>` to expand VRAM capacity dynamically.");
            }
        }
        Commands::ProbeDmaBuf => {
            println!("=== OpenMemGPU Linux DRM / Vulkan DMA-BUF Feasibility Probe ===");
            let report = openmemgpu_transfer::DmaBufProbe::probe()?;
            for dev in &report.devices {
                println!("\nDevice: {}", dev.device_name);
                println!("  PCI Address: {:?}", dev.pci_address);
                println!(
                    "  VK_KHR_external_memory_fd: {}",
                    dev.supports_external_memory_fd_extension
                );
                println!("  DMA-BUF Exportable: {}", dev.dma_buf_exportable);
                println!("  DMA-BUF Importable: {}", dev.dma_buf_importable);
                println!(
                    "  Compatible Handle Types: {:#x}",
                    dev.compatible_handle_types
                );
                println!("  Details: {}", dev.findings);
            }
            println!("\nOverall Evaluation:");
            println!(
                "  Direct Cross-GPU DMA Supported: {}",
                report.direct_cross_gpu_dma_supported
            );
            println!("  Summary: {}", report.architectural_summary);
        }
    }

    Ok(())
}
