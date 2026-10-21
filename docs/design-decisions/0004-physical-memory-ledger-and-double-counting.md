# ADR 0004: Physical Memory Ledger & Prevention of Double-Counting

- **Status**: Accepted
- **Deciders**: OpenMemGPU Systems Architecture Team
- **Date**: October 21, 2026

## Context
The AMD Renoir integrated GPU and the Navi 10 discrete GPU's GTT heap both allocate physical storage from the same 14.0 GiB DDR4 system RAM pool. Summing the reported capacities of each heap would report over 28 GB of memory on a 14 GB system.

## Decision
VVRAM must maintain a centralized `PhysicalMemoryLedger`. All allocations requested from iGPU heaps or dGPU host-visible GTT heaps must debit against the single global `HostPhysicalRam` budget pool.

## Consequences
- Guaranteed mathematical correctness in capacity reporting.
- Elimination of out-of-memory kernel panics or Linux OOM killer invocations.
- Honest, scientifically reproducible capacity metrics.
