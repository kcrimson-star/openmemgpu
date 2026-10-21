# Workload Scheduler & Multi-GPU Coordination Design

**Document Version**: 1.0  
**Status**: Architecture Baseline  

---

## 1. Scheduling Principles on Heterogeneous Hardware

In a homogeneous cluster (e.g. 4x NVIDIA RTX 4090), workload partitioning is symmetric (each GPU receives 25% of the batch or layers).

In a mobile heterogeneous platform like our AMD Advantage system, computing capabilities are severely asymmetric:
- **dGPU (Navi 10)**: 36 Compute Units, ~7.0 TFLOPS FP32, 288 GB/s memory bandwidth.
- **iGPU (Vega 6)**: 6 Compute Units, ~1.1 TFLOPS FP32, ~35 GB/s memory bandwidth.

**Amdahl's Law Warning**: If a parallel workload is split 50/50 across both GPUs, the dGPU finishes its half in 14% of the time and sits completely idle for the remaining 86% of the cycle waiting for the iGPU!

```
Symmetric Split (50 / 50) - FAILS:
dGPU: [======= COMPUTE =======] [ ---------------- IDLE / WAITING ---------------- ] [ BARRIER ]
iGPU: [================================= COMPUTE =================================] [ BARRIER ]

Asymmetric Split (Weighted by Compute Ratio ~86% dGPU / 14% iGPU) - BALANCED:
dGPU: [============================= COMPUTE =============================] [ BARRIER ]
iGPU: [============================= COMPUTE =============================] [ BARRIER ]

Pipelined Layer Execution (Model Slicing):
dGPU (Layers 0..24): [ Layer 0..24 ] ---> [ Layer 0..24 ] --->
iGPU (Layers 25..27):                      [ Layer 25..27] ---> [ Embed/Head ]
```

---

## 2. Supported Scheduling Paradigms

OpenMemGPU implements three execution modes:

### 2.1 Mode 1: Discrete Primary with APU Offloading (Capacity Expansion)
- The primary workload (e.g. transformer attention layers) runs on the discrete GPU.
- When model size exceeds dGPU VRAM (e.g. a 7B or 13B quantized model requiring 8-10 GB), intermediate layers or less compute-intensive layers (embeddings, output projections, KV cache history) are scheduled on the iGPU or hosted in system RAM.

### 2.2 Mode 2: Asymmetric Workload Partitioning
- For data-parallel operations (e.g. matrix multiplications, vector dot-products, batch processing), the scheduler computes a capacity weight:
  $$W_{\text{dGPU}} = \frac{\text{TFLOPS}_{\text{dGPU}}}{\text{TFLOPS}_{\text{dGPU}} + \text{TFLOPS}_{\text{iGPU}}} \approx \frac{7.0}{7.0 + 1.1} \approx 0.864$$
  $$W_{\text{iGPU}} = 1 - W_{\text{dGPU}} \approx 0.136$$
- Task chunks are sized proportional to device throughput to ensure synchronized completion without barrier stalling.

### 2.3 Mode 3: Pipelined Streaming
- For sequential neural networks, execution is arranged as a pipeline where device transfers occur concurrently with compute passes on the other device using double-buffered staging queues.
