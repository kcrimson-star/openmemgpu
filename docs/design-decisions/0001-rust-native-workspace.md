# ADR 0001: Rust-Native Workspace Architecture

- **Status**: Accepted
- **Deciders**: OpenMemGPU Systems Architecture Team
- **Date**: October 21, 2026

## Context
OpenMemGPU requires a robust systems programming language providing low-level memory control, zero-cost abstractions, fearless concurrency, fine-grained FFI boundary safety, and a first-class package management system.

## Decision
All project-owned implementation code for OpenMemGPU and VVRAM will be authored exclusively in Rust. Necessary native dependencies (e.g. Vulkan loader `libvulkan.so.1`, OS APIs) will be interfaced through safe Rust abstractions and audited, minimal `unsafe` blocks.

## Consequences
- Guarantees memory safety and data race protection across multi-threaded and multi-queue GPU orchestration pipelines.
- Standardizes tooling around Cargo, Clippy, and Rustfmt.
- Strict isolation of low-level FFI invariants.
