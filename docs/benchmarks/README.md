# OpenMemGPU & VVRAM Benchmark Framework

**Document Version**: 1.0  
**Status**: Benchmark Methodology Specification  

---

## 1. Governing Benchmark Methodology

To ensure absolute scientific reproducibility and prevent exaggerated or fabricated claims, every benchmark in OpenMemGPU must adhere to these five rules:

1. **Explicit Evidence Labeling**: Every published metric must carry one of these strict labels:
   - `[Measured on Physical Hardware]`
   - `[Measured in Virtualized Environment]`
   - `[Simulated]`
   - `[Estimated]`
   - `[Not Yet Measured]`
2. **Standard Baselines**: All multi-GPU or VVRAM tests must be benchmarked against single-device baselines:
   - Baseline 1: CPU-only execution.
   - Baseline 2: iGPU-only execution (Radeon Vega 6).
   - Baseline 3: dGPU-only execution (Radeon RX 5600M).
   - Baseline 4: Concurrent independent execution across both GPUs.
   - Target: VVRAM-managed execution.
3. **Warm-up and Repetition**: Every benchmark must execute a minimum of 5 warm-up iterations followed by at least 20 recorded runs, reporting median, 95th percentile, and standard deviation.
4. **Machine-Readable Outputs**: All benchmark harnesses must output raw JSON telemetry files in `docs/benchmarks/results/` alongside human-readable markdown summaries.
5. **No Synthetic Extrapolation**: Never present an estimate as a measured result.

---

## 2. Core Benchmark Suites (`openmemgpu-bench`)

| Suite Name | Target Metrics | Key Measurements |
| :--- | :--- | :--- |
| `bench_alloc_latency` | Memory Allocation & Deallocation | Latency to allocate and bind 1 MiB to 1 GiB buffers; time to query memory budgets. |
| `bench_transfer_bandwidth` | Interconnect Throughput | Host-to-Device (H2D), Device-to-Host (D2H), and Device-to-Device (D2D) bandwidth in GB/s. |
| `bench_timeline_sync` | Synchronization Overhead | Latency of `VK_KHR_timeline_semaphore` cross-device signaling in microseconds. |
| `bench_compute_throughput` | Matrix Multiplication / Vector FP32 | TFLOPS achieved on dGPU vs iGPU vs partitioned concurrent dispatch. |
| `bench_vvram_overcommit` | Memory Overcommitment & Eviction | Maximum model/tensor size allocatable before OOM; eviction paging latency under pressure. |
| `bench_ai_inference` | AI / LLM Layer Inference | Time to first token (TTFT), tokens per second (TPS), peak resident VRAM per device. |

---

## 3. Benchmark Execution Command

Once the Rust toolchain is compiled:
```bash
# Run the complete hardware benchmark suite and export JSON reports
cargo run --release -p openmemgpu-cli -- bench --all --output docs/benchmarks/results/latest.json
```
