# AetherDCC Engine v2.0

`aether-dcc-engine` is a high-performance, zero-dependency Windows Rust runtime for non-binary silicon compute and ternary AI acceleration.

One cohesive workspace unifies a **quaternary contention CPU ALU**, a **Universal Direct3D 11 1.58-bit BitNet digit inference pipeline**, the **GhostVault SIMD cryptographic accelerator**, the **Kleene/Priest paradox isolator**, and a **64-node Ising spin-glass relaxer**. It exports a thread-safe C ABI, an interactive Win32 studio, and a verifiable release scorecard.

---

## What's New in Engine v2.0 (Product Re-Engineering)

- **Universal GPU Portability:** Removed hardcoded Intel UHD `0x9B41` restrictions. The Direct3D 11 compute engine dynamically enumerates all DXGI adapters (Intel, NVIDIA, AMD, Qualcomm Snapdragon) and automatically selects the highest-performing compute adapter with full feature level 11.0/11.1 support.
- **Chopper Denominator Gated Sentinel:** Enhanced the continuous-timing telemetry classifier with the White Paper's *Chopper Denominator Gate* (`min_energy * 3.0x`), filtering asymmetric OS thread preemption spikes and guaranteeing rock-solid, deterministic classification.
- **Clean MSVC Linker Target Separation:** Resolved Cargo PDB symbol collisions by cleanly isolating the library (`aether_dcc.dll` / `.lib`) from the studio binary (`aether-studio.exe`).
- **Verilog RTL Co-Simulation:** Added software twin unit tests verifying cycle-accurate bit-for-bit equivalence between the Rust CPU ALU and the synthesizable `rtl/dcc_arbiter.v` hardware core.
- **Extended CLI Diagnostic Suite:** Added CLI commands for GPU hardware enumeration (`--info`), rapid quaternary ALU verification (`--test-alu`), and headless release benchmarking (`--benchmark`).
- **Zero Warnings & Standard C Header:** 100% warning-free compilation under `rustc 1.97+`, accompanied by an official C/C++ header (`include/aether_dcc.h`).

---

## Requirements

- **Operating System:** Windows x86_64.
- **Rust Toolchain:** Rust 2021 edition (`rustc` / `cargo` 1.80+).
- **GPU:** Any DirectX 11.0+ compatible GPU (Intel Iris/UHD/Arc, NVIDIA GeForce/RTX, AMD Radeon, or Qualcomm Adreno).
- **Dependencies:** **Zero crates.** Links `kernel32`, `user32`, `gdi32`, `d3d11`, `dxgi`, and `d3dcompiler_47` purely via raw FFI.

---

## Quickstart

### Build
```cmd
cargo build --release
```

### Run Hardware Diagnostics
```cmd
cargo run --release -- --info
```

### Run Executive Product Benchmark
```cmd
cargo run --release -- --benchmark
```

### Launch Interactive Win32 Studio GUI
```cmd
cargo run --release
```

### Run Full Test Suite
```cmd
cargo test --release -- --test-threads=1
```

---

## Architecture

- **Quaternary CPU ALU:** One sensor thread and 3 synapse threads execute `lock xadd` on a 4096-byte `VirtualAlloc` page. Channels are spaced with 256-byte strides, placing positive and negative atomic rails on distinct 64-byte cache lines to eliminate spatial prefetch collisions. Pedestal 4 and burst 4 produce stabilized plateaus at $-0.600$, $-0.200$, $+0.200$, and $+0.600$, decoded into magnitudes $-3, -1, +1, +3$.
- **Common-Mode Cancellation:** Given thermal and frequency drift scaling $\alpha(T, f)$:
  $$Y = \frac{\alpha t_{\text{pos}} - \alpha t_{\text{neg}}}{\alpha t_{\text{pos}} + \alpha t_{\text{neg}}} = \frac{t_{\text{pos}} - t_{\text{neg}}}{t_{\text{pos}} + t_{\text{neg}}}$$
  The shared scale divides out identically.
- **DCC-BitNet:** Direct3D 11 compute shader (`cs_5_0`) maps 64 pixels through 16 ternary hidden neurons to 10 digit classes. Zero weights skip `InterlockedAdd` entirely (81.08% of weights), saving bus bandwidth and energy.
- **Hardware Digital Twin:** `rtl/dcc_arbiter.v` provides a multiplier-less, divider-less synthesizable Verilog-2001 arbiter using shift-add comparator networks.

---

## C ABI

The build produces `target/release/aether_dcc.dll`. An official header is provided in [`include/aether_dcc.h`](include/aether_dcc.h).

```c
#include "aether_dcc.h"

// Version query
const char *ver = dcc_sdk_version();

// 1.58-bit Neural Inference
float pixels[64];
int class_out, scores[10];
double latency_us;
dcc_sdk_infer_digit10(pixels, &class_out, scores, &latency_us);

// 64-node Ising Spin-Glass Solver
int spins[64], energy;
dcc_sdk_solve_ising64(neighbors, coupling, bias, spins, &energy, &latency_us);
```

---

## License

MIT License. Copyright (c) 2026 Agaran / AetherDCC Systems.
