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
| R5 | `rockfish-hip`: safe HIP layer, design in `docs/DESIGN-R5.md`. R5a (error, devices) merged; R5b (streams, events, buffers, optional `half`) merged; R5c (modules, kernel launch) merged, launch path UNVERIFIED (no code object to run); R5d `rockfish-hiprtc` merged, UNVERIFIED (no hipRTC on the laptop); R5e pinned memory and async copies merged (round trip measured, one device) | merged (open item: R5-H, next row) |
| R5-H | Finding, UNVERIFIED beyond one machine: parallel first use of HIP hung; see section R5-H below | warm-up in R5b; root cause open: needs a second AMD machine (board 329) |
| R6 | Math libraries: rocBLAS, hipBLASLt, rocFFT, rocRAND, rocSOLVER, rocSPARSE, MIOpen. R6a `rockfish-rocblas-sys` merged (56 functions; symbols and version measured against Ollama's ROCm 7.1 `rocblas.dll`, no GPU call yet). R6b `rockfish-hipblaslt-sys` merged (whole C API, 24 functions, all exported by Ollama's `libhipblaslt.dll`). R6c `rockfish-miopen-sys` merged (42 functions, conv/tensor/activation; chosen from Fuel's source: its CUDA backend uses cuDNN for Conv1D/Conv2D, and cuRAND only for a path its RNG-seam spec replaces; UNVERIFIED, no MIOpen library on the laptop). R6d safe `rockfish-rocblas` merged (typed GEMM, strided batched, f32/f64/f16/bf16; `version()` measured, GEMM UNVERIFIED on a GPU). R6e `rockfish-rocfft-sys` (29 functions, the public C API; header verified byte-identical to upstream; no rocFFT library on the laptop, so symbols and library names UNVERIFIED). R6f `rockfish-rocsolver-sys` (36 curated s/d dense factorisation and solver functions on the rocBLAS types; header byte-identical to upstream; symbols and library names UNVERIFIED, no rocSOLVER library on the laptop). R6g `rockfish-rocsparse-sys` (84 curated s/d functions incl. the generic spmv/spmm/spgemm/sddmm API; 113 headers byte-identical to upstream; symbols and library names UNVERIFIED, no rocSPARSE library on the laptop). R6h `rockfish-rocrand-sys` (30 host-API functions; 4 headers byte-identical to upstream; the half-precision generators are not bound; symbols and library names UNVERIFIED, no rocRAND library on the laptop). R6 -sys list complete. Next: (queued, below consumer needs; no Fuel or Bonefish use found: Bonefish's FFT is oneMKL DFT). R6-L merged: `rockfish-core` now loads an absolute-path DLL with its own directory searched for dependencies (Ollama's bundle loads without `PATH`; measured on one laptop) | merged; rocFFT sys in review (#50); rocSOLVER sys on a branch; rocSPARSE and rocRAND sys on a branch; below consumer needs |
| R7 | HIP C++ parser + emitter on Unpopped's `Frontend`/`Backend` contract (`Unpopped/docs/parser-emitter-ownership-map.md`, ROADMAP 6.7); design in `docs/DESIGN-R7.md` (R7a parse, R7b emit, R7c reductions/scans, R7d hipRTC compiler); nothing compiled or run on a GPU | R7a `rockfish-hip-parse`, R7b `rockfish-hip-emit` (f32 scalar elementwise), R7c (sequential reduction + serial scan; f64; f16/bf16 scalar + vectorized elementwise via the software codec, 0.0.35; integer sequential Sum/Prod/Max/Min reduction, 0.0.36; integer serial Sum/Prod/Max/Min scan, 0.0.37; float sliding-window pool (Sum/Mean/Prod/Max/Min, any axis), 0.0.38; strided elementwise (transposed/broadcast operands, Coord; f32/f64), 0.0.39; integer strided elementwise (no Coord), 0.0.40; integer elementwise Max/Min and the Unpopped 0.45.17 repin, 0.0.41; Unpopped 0.47.0 repin (walkers now in `unpopped-convert`), 0.0.42; Unpopped 0.56.2 repin and the by-name refusal of NVIDIA-only kernel statements, 0.0.45; integer sliding-window pool (Sum/Prod/Max/Min), 0.0.46; f16/bf16 strided elementwise and sliding-window pool, 0.0.47; window post-expression over the folded value and the Unpopped 0.56.5 repin, 0.0.48; total integer `Div`/`Shl`/`Shr`/`ConstInt` at i32/i64/u32/u64 scalar; integer comparisons + `Select` at i32/i64/u32/u64 so `Rem` composes (Unpopped 0.40.11); host-compiled differential check on the boundary cross product, no HIP compiler), R7d `HipRtcCompiler` and the `Vectorized` elementwise schedule (float2/float4) all merged; compile of any emitted text UNVERIFIED (rig row) |
| R7-CI | CI honesty (pm-kernels TASK 2026-11-23): hip-emit `gather`, `int_run`, `syntax` printed a skip and passed when no `clang++` was found. `REQUIRE_CLANG=1` now makes that a failure (`tests/common`); `ci.yml` sets it and installs clang on ubuntu. Red-first shown on one machine (clang hidden, REQUIRE_CLANG=1: FAILED; unset: ok; real clang: ok). Test matrix os list: ubuntu-latest, windows-latest, macos-latest. UNVERIFIED: that windows-latest and macos-latest resolve `clang++` on PATH (first CI run reads it) | in PR (next window 11-24) |
| R7-SUP-001 | CodeRipper SUP-001 (Medium): `cargo` runs without `--locked` at 3 sites, `.github/workflows/ci.yml:36` and two more. Source: PR #54 advisory `portfolio` job at 467fd259 (run 45526384302) | done on `lane/ci-locked-gov010`, PR tomorrow's first |
| R7-SUP-016 | CodeRipper SUP-016 (Medium): 91 of 92 source files carry no SPDX-License-Identifier in the first 15 lines (first hit `crates/rockfish-core/src/lib.rs:1`). Add the header; keep the vendored/generated files' upstream notices as they are. Source: same run | done on `lane/ci-locked-gov010` (gate exit 0 on the committed tree) |
| R7-GOV-010 | CodeRipper GOV-010 (Low x119): `#[allow(...)]` without `reason =`, across 20 files: 103 in the generated `*-sys/src/generated.rs` files (81 in `rockfish-hip-sys`; fix in the generator, `gen-bindings`) and 16 hand-written (`*-sys/src/lib.rs`, `rockfish-hip-emit` `src/lib.rs:1202` and `tests/common/mod.rs:7`, `rockfish-rocblas`, `rockfish-hiprtc`). Same one-pass `reason =` pattern as Unpopped's 54. Source: same run | done on `lane/ci-locked-gov010`: 16 hand-written + 103 generated (the generator adds the `reason =`) |
| R-PUBLISH-NOTICES | Publish gate (pm-kernels FIX 2026-11-25): the `*-sys` crate dirs hold only Cargo.toml, src, tests (no LICENSE-*, no THIRD-PARTY-NOTICES.md, no `include`/`exclude`), and the notices live at the repo root, which a published `.crate` does not contain, so the pointer in each `generated.rs` would not travel. Before the first publish of any `-sys` crate: `cargo package --list -p <crate>` shows LICENSE-MIT, LICENSE-APACHE and the notices file the generated.rs points to (the crate dir carries synced copies, with a CI check that they match the root, as fuel does), and rocsolver's package carries the BSD-2-Clause text. Nothing is published yet (`rockfish` 0.0.0 from 10-30), so not a live defect | DONE on `lane/publish-files`: each of the 17 crate dirs carries LICENSE-MIT (LICENSE-APACHE x17 = 196k diff chars, over the 150k PR limit: follow-up PRs, OPEN), the 9 `generated.rs` crates carry a THIRD-PARTY-NOTICES.md holding only their own header section, rocsolver-sys carries LICENSE-BSD-2-Clause; `tools/check-synced-copies.sh` (run in the vendor CI job) fails on drift; `tools/check-package-files.py` is the proof (all 17 OK). `publish = false` everywhere, so no published crate was affected |
| R8 | Fuel ROCm backend built on Rockfish (design with the Fuel lane; Baracuda is the precedent) | after R5 |

## R7-D: kernel statements `Hip` declines, scheduled and not skipped (CIRESNAVE-EXPECTATIONS 6.21; 2026-11-27, Unpopped 0.60.8 fixtures)

One row per Unpopped fixture the HIP emitter refuses by name (13 with a patent, 8 with "no lowering"; `tests/fixture_refusal.rs`). A row is **scheduled a way down, not declined for good**: the last column is what would unblock it. Nothing here is a clearance.

| Row | Statement | Why declined | What would unblock it |
|---|---|---|---|
| R7-D01 | `Collective` | US8539204B2, board 434 C; board 506 A lifted only subgroup shuffle/vote/reduce/broadcast, lowered by `lower_kernel_for` | subgroup scans (US9417875B2), block-scope collectives, float reductions, GCN3/Vega/CDNA5 ids: each needs a ruling, a patent read or a hardware page |
| R7-D02 | `CollectiveTuple` | US8539204B2, board 434 C; 506 A did not name tuple-state collectives | a ruling on tuple collectives and a module-level lowerer |
| R7-D03 | `MbarInit` | US11803380B2 (mbarrier), CireSnave 2026-11-16 (board 453) | a patent read and ruling; whether AMD has an equivalent object is unread |
| R7-D04 | `MbarArrive` | US11803380B2, board 453 | as MbarInit |
| R7-D05 | `MbarCompleteTx` | US11803380B2, board 453 | as MbarInit |
| R7-D06 | `MbarWait` | US11803380B2, board 453 | as MbarInit |
| R7-D07 | `WgmmaWait` | US12204897B2 and US12710961B2, CireSnave 2026-11-16 | a patent read and ruling; whether AMD MMA needs a wait is unread (see board 519 B) |
| R7-D08 | `Tcgen05Commit` | US11803380B2 by the PM's extension (not read separately against the claims) | the PM's extension read against the claims, then a ruling |
| R7-D09 | `CopyAsync` | US12118382B2 and US11294713B2 (async data movement), board 434 C | a patent read and ruling; an AMD async-copy mechanism is unread |
| R7-D10 | `CopyBulk` | US11803380B2 (a copy on an mbarrier), board 453 | as MbarInit; tensor-map family US12657024/US12669996/US12645455 also unread |
| R7-D11 | `CopyUnpack` | US11803380B2 (Unpopped 0.58.0 addition) | as CopyBulk |
| R7-D12 | `AsyncCommit` | US12118382B2 and US11294713B2, board 434 C | as CopyAsync |
| R7-D13 | `AsyncWait` | US12118382B2 and US11294713B2, board 434 C | as CopyAsync |
| R7-D14 | `ProxyFence` | no patent claim; no AMD counterpart decided (an NVIDIA async-proxy fence) | decide whether an AMD fence (`__threadfence_block`/`__threadfence`, not read here) is a faithful lowering; needs the memory-model read |
| R7-D15 | `Wgmma` | no patent claim made; an NVIDIA warpgroup MMA (a_layout/b_layout included) | lowering to AMD MFMA/WMMA through the opaque-tile statements Unpopped's B1 release adds (board 519 B, CireSnave's risk call, not a clearance); do not pre-build |
| R7-D16 | `WgmmaCommit` | no patent claim made; an NVIDIA wgmma group commit | as Wgmma; whether AMD needs a commit is unread |
| R7-D17 | `Tcgen05Mma` | no patent claim made; an NVIDIA tensor-memory MMA | as Wgmma; whether AMD has a tensor-memory analogue is unread |
| R7-D18 | `Tcgen05Alloc` | no patent claim made; an NVIDIA tensor-memory allocation | whether AMD has a tensor-memory analogue is unread; unblock by that read and a design, or a decision to keep it NVIDIA-only |
| R7-D19 | `Tcgen05Dealloc` | no patent claim made; an NVIDIA tensor-memory release | as Tcgen05Alloc |
| R7-D20 | `Tcgen05RelinquishPermit` | no patent claim made; an NVIDIA allocation-permit relinquish | as Tcgen05Alloc |
| R7-D21 | `Tcgen05Ld` | no patent claim made; an NVIDIA tensor-memory load | as Tcgen05Alloc |

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
