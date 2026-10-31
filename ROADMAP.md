# Rockfish roadmap

Order matters: survey, then design, then code. Every code row ships with a negative control (a test that fails when the behaviour is removed). Linux and Windows from the start; anything not measured on real AMD hardware is labelled unmeasured (there is no ROCm test box yet).

| Row | What | Status |
|---|---|---|
| R0 | Workspace skeleton, CI (fmt, clippy, test x3 OS, doc, cargo-deny, portfolio), CODEOWNERS | merged |
| R1 | Branch protection requiring the R0 checks (PM) | merged (PM) |
| R2 | `docs/SURVEY.md`: existing crates (rocm-rs, rocmrc, cubecl-hip-sys, rocm-sys), ROCm/HIP SDK library names per OS, what loads on Windows; options first, recommendation, no code | merged |
| R3 | `docs/DESIGN.md`: sys/safe split, runtime loading, per-OS library table, error model, Fuel consumption (with the Fuel lane) | merged |
| R3b | Spike: HIP version / struct-layout handling; result in `docs/SPIKE-R3b-versions.md` | merged |
| R4a | `rockfish-core`: loader, per-OS name table, error type, tests against a real system library | merged |
| R4b-1 | Vendor the HIP headers with provenance and a byte-for-byte check | merged |
| R4b-2 | `rockfish-hip-sys` and `rockfish-hiprtc-sys`: bindgen-generated HIP runtime + hipRTC bindings on the R4a loader. Library names other than `amdhip64_6.dll` (measured, one laptop) are UNVERIFIED | merged |
| R5 | `rockfish-hip`: safe HIP layer, design in `docs/DESIGN-R5.md`. R5a (error, devices) merged; R5b (streams, events, buffers, optional `half`) merged; R5c (modules, kernel launch) merged, launch path UNVERIFIED (no code object to run); R5d `rockfish-hiprtc` merged, UNVERIFIED (no hipRTC on the laptop); R5e pinned memory and async copies merged | in progress |
| R5-H | Finding, UNVERIFIED beyond one machine: parallel first use of HIP hung; see section R5-H below | warm-up in R5b; root cause open |
| R6 | Math libraries: rocBLAS, hipBLASLt, rocFFT, rocRAND, rocSOLVER, rocSPARSE, MIOpen. R6a `rockfish-rocblas-sys` merged (56 functions; symbols and version measured against Ollama's ROCm 7.1 `rocblas.dll`, no GPU call yet). R6b `rockfish-hipblaslt-sys` merged (whole C API, 24 functions, all exported by Ollama's `libhipblaslt.dll`). R6c `rockfish-miopen-sys` in PR (42 functions, conv/tensor/activation; chosen from Fuel's source: its CUDA backend uses cuDNN for Conv1D/Conv2D, and cuRAND only for a path its RNG-seam spec replaces; UNVERIFIED, no MIOpen library on the laptop). Next: safe `rockfish-rocblas`; rocFFT, rocRAND, rocSOLVER, rocSPARSE have no Fuel use found in its source at origin/main, so they wait for a request. Follow-up R6-L: `rockfish-core` loads by absolute path without the DLL's own directory, so a library whose dependencies sit beside it (Ollama's bundle) fails unless that directory is on `PATH` | in progress |
| R7 | HIP C++ parser + emitter on Unpopped's `Frontend`/`Backend` contract (`Unpopped/docs/parser-emitter-ownership-map.md`, ROADMAP 6.7) | after R3 |
| R8 | Fuel ROCm backend built on Rockfish (design with the Fuel lane; Baracuda is the precedent) | after R5 |

## R5-H: parallel first use of HIP hung on one machine

**Measured on:** Windows laptop, `amdhip64_6.dll` (driver bundled, HIP 6.4.50501), Radeon 610M (gfx1036), one device. Nothing else.

**What happened:** with the `rockfish-hip` GPU tests running in parallel (the default), the test process hung until killed (90 s timeout; every parallel run, about 8 of 8 across variants). With `--test-threads=1` the suite passed in under 2 s.

**Narrowed to:** one thread's first `hipStreamCreate` overlapping other threads' first HIP calls. A test that only creates a stream, run beside a test that only queries the device (`hipSetDevice`, `hipDeviceGetName`, `hipMemGetInfo`), hung; the other test pairs I ran passed.

**Repro (before the warm-up):** in `crates/rockfish-hip/src/lib.rs` remove the `warm_up(&api);` call in `hip()`, then `cargo test -p rockfish-hip --lib` hangs, while `cargo test -p rockfish-hip --lib -- --test-threads=1` passes. To run just the pair, build the test binary and run it with two test names: `rockfish_hip-<hash>.exe events_record_complete device_queries`.

**What the warm-up changes:** `hip()` creates and destroys one stream on device 0 under a process-wide lock before any thread gets the table. With it, 5 of 5 parallel runs passed.

**Tried and did NOT fix it:** `hipInit(0)` under the lock; `hipSetDevice(0)` plus `hipFree(null)` under the lock; non-blocking streams (`hipStreamNonBlocking`).

**NOT tested:** other devices, or a second device on the same machine; other driver or HIP versions; Linux or any ROCm install; discrete GPUs; whether only the first `hipStreamCreate` matters or other first calls do too; whether the device-0 warm-up is enough when another device is used first.

**What this is not:** not evidence that HIP is not thread-safe. It is one hang on one runtime, with a mitigation that worked on that runtime and a cause nobody has found.

**Next:** repeat on any other AMD machine or driver that appears (the self-hosted runner question, board item 329) and report whether the hang reproduces without the warm-up.
