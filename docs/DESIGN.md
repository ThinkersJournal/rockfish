# Rockfish design (R3)

Design only, no crate code. Builds on `docs/SURVEY.md` (R2). The PM ruled option B on 2026-10-30: our own sys crates, runtime-loaded, per-OS library names in one place; `cubecl-hip-sys` stays a reference. Claims are labelled **[measured]**, **[read]** (source or docs read, nothing built) or **[unverified]**. There is no ROCm test box yet.

## 1. Crate layout (Baracuda's shape)

Baracuda ships one `-sys` and one safe crate per library plus parse/emit crates **[read]**: `baracuda-cublas-sys` + `baracuda-cublas`, `baracuda-cuda-parse`, `baracuda-cuda-emit`, and so on, with Fuel depending on the sub-crates directly rather than the umbrella **[read]** (Fuel `Cargo.toml` at `origin/main` 750cd2fc3, per the Fuel lane). Rockfish copies the shape:

| Crate | Role | Wave |
|---|---|---|
| `rockfish-core` | library loader, per-OS name table, error type, version type | 1 |
| `rockfish-hip-sys` | HIP runtime bindings, runtime-loaded | 1 |
| `rockfish-hiprtc-sys` | hipRTC bindings | 1 |
| `rockfish-hip` | safe layer: devices, contexts, streams, events, device memory, modules, kernel launch | 1 |
| `rockfish-hiprtc` | safe runtime compile | 1 |
| `rockfish-rocblas[-sys]`, `rockfish-hipblaslt[-sys]` | GEMM | 2 |
| `rockfish-rocrand[-sys]`, `rockfish-miopen[-sys]`, `rockfish-rocfft[-sys]`, `rockfish-rocsolver[-sys]`, `rockfish-rocsparse[-sys]` | rest | 3 |
| `rockfish-hip-parse`, `rockfish-hip-emit` | HIP C++ parser and emitter on Unpopped's `Frontend`/`Backend` contract | after wave 1 |
| `rockfish` | umbrella re-exports | with wave 1 |
| `rockfish-sys` | already reserved at 0.0.0; becomes an umbrella over the `-sys` crates (or stays a placeholder: decided at first release) | with wave 1 |

`rockfish-core` is new relative to Baracuda only if Baracuda has no equivalent; I have not compared (`baracuda-core` exists). One version number across the whole workspace (portfolio rule), allocated by the PM.

## 2. Loading

- Every `-sys` crate binds through a runtime loader in `rockfish-core` (`libloading`, MIT/ISC, latest version). Nothing links at build time; building needs no ROCm.
- Every binding is a function pointer in a table filled lazily. A library that cannot be found is `Err(LoadError::LibraryNotFound { tried })`; a symbol that is missing is `Err(LoadError::SymbolMissing { name })` **for that call only**. No panics (contrast `cubecl-hip-sys`, which panics **[read]**).
- **Library names live in one table** in `rockfish-core` (`LibrarySpec { logical_name, windows: &[..], linux: &[..] }`). Search order: explicit path from the caller, `ROCFISH_<LIB>_PATH`, `HIP_PATH`/`ROCM_PATH` (`bin` on Windows, `lib` on Linux), then the default loader path.
- **Windows runtime-DLL requirement [measured on the laptop iGPU only]:** the driver-bundled runtime is `C:\Windows\System32\amdhip64_6.dll` (HIP 6.4.50501, Radeon 610M gfx1036); it loads, allocates and copies, and there is no hipRTC DLL beside it. So the HIP entry in the table must list versioned names, newest first (`amdhip64_7.dll`, `amdhip64_6.dll`, then `amdhip64.dll`), and the safe layer must treat **hipRTC as optional**. Evidence is one machine with one iGPU; names for other HIP releases and for the SDK's DLLs are **[unverified]**.
- Linux names (`libamdhip64.so`, `.so.7`, `.so.6`, ...) and the rest of the Windows names are **[unverified]** until read from a real install; the table starts with what `rocmrc` and `cubecl-hip-sys` try **[read]** and is corrected by measurement.

## 3. Versions

HIP bindings differ between releases, but less than feared. Spike R3b (`docs/SPIKE-R3b-versions.md`, **[read]** from the 11 bindgen files in `cubecl-hip-sys` 7.14.6085001) found `hipDeviceProp_tR0600` identical in all 11 HIP builds, 88 functions added and 1 removed over the range, signature changes ABI-neutral (const-ness, `c_int` to `c_uint`), and layout changes only in rare structs (`hipMemAllocationProp`, `hipLaunchAttributeValue`). Rockfish's rule: generate bindings from the **newest** headers and resolve every symbol per call, so a symbol absent from an older runtime returns `SymbolMissing` for that call only. The HIP version is read at run time from `hipRuntimeGetVersion` (**[measured]** on the laptop: 60450101) and exposed as `rockfish_hip::runtime_version()`; rare structs with layout changes are fenced behind a version check and added only on demand. Not covered by the spike: ROCm before this range and the math libraries (wave 2).

## 4. Platform support and test matrix

| Where | GPU | What can run | Status |
|---|---|---|---|
| Laptop, Windows | Radeon 610M gfx1036 | HIP runtime: device query, malloc, copy | **[measured]**; kernels not measured (no hipRTC) |
| Desktop, Windows | Radeon VII gfx906, Arc, NVIDIA | HIP SDK for Windows | AMD's Windows HIP SDK table (page labelled 7.2.0, "legacy") lists gfx906 (Radeon PRO VII) as **not supported**, and lists only gfx1100/1101/1102/1150/1151/1200/1201 as supported **[read, 2026-10-30]**. Expect Radeon VII and MI50 to be Linux-only; **[unverified]** by measurement. |
| Desktop, Linux | Radeon VII, MI50 (on order) | full ROCm | no box yet; ROCm's own gfx906 support status in the current release is **[unverified]** and must be read before promising anything |
| CI hosted runners | none | build, unit tests with a stub library, loader error paths | all three OSes |

Rules: tests that need a GPU run on the self-hosted runner under the portfolio rule, never `#[ignore]`d; everything else is testable without a GPU through a fake shared library that `rockfish-core`'s own tests build, so the loader's error paths (library missing, symbol missing, wrong version) have negative controls in ordinary CI.

## 5. Error model

One `rockfish_core::Error` per layer, `#[non_exhaustive]`, carrying the HIP status code **and its name** (`hipGetErrorName` when loadable, a built-in table when not) plus the call that failed. Load errors and API errors are distinct variants. No panics from FFI wrappers; `unsafe` is confined to `-sys` crates and the few safe-layer spots that hold raw handles, each with a `// SAFETY:` comment.

## 6. What Fuel needs (first ROCm ops)

From the Fuel lane (message 2026-10-30, `origin/main@750cd2fc3`, **inference not measured on ROCm**): Fuel's CUDA backend consumes `baracuda-types`, `-core`, `-cuda-sys`, `-driver` (Context, Stream, DeviceBuffer, DeviceSlice), `-runtime`, `-cublas` (strided-batched GEMM), `-cudnn`, `-curand`, `-nvrtc`, `-nccl`, `-cutlass`, `-cuda-seam` and hand kernels, and binds each op through an FKC contract. First ops for a ROCm backend, from the Qwen3.8-27B decode path: dense and batched GEMM (f16/bf16/f32, accumulate f32), elementwise unary/binary/cast, reduce/softmax/rmsnorm, gather/index_select, write_slice, rope, decode attention, quantized matvec (Q4_K/Q6_K/Q8_0). Dtypes f32/f16/bf16, u32 ids, u8 quant blocks. Consequences for wave order: `rockfish-hip` + `rockfish-hiprtc` + rocBLAS/hipBLASLt come first; kernels for the non-GEMM ops are HIP C++ source compiled with hipRTC, which is also the first customer of the R7 emitter. Mapping Rockfish types to Fuel's backend contract is a joint design with the Fuel lane; Fuel's repo is not edited from here. The Fuel lane also notes that the MI50 and Radeon VII are reached through Fuel's Vulkan backend today.

## 7. HIP C++ parser and emitter

Owned by Rockfish (CireSnave, 2026-10-30, via `Unpopped/docs/parser-emitter-ownership-map.md` section 9). Precedent: `baracuda-cuda-parse` and `baracuda-cuda-emit`. Scope and grammar subset are a separate design (R7 starts with reading Unpopped's `Frontend`/`Backend` traits at its current `origin/main`, not here). HIP C++ is close to CUDA C++, so the first emitter target is the subset Fuel's kernels need, and the parser's first input is the same subset.

## 8. Rows this changes in ROADMAP

R4 becomes `rockfish-core` + `rockfish-hip-sys` + `rockfish-hiprtc-sys` with the fake-library tests; R5 the safe layer. The version/struct-layout spike (R3b) is done, see section 3.

## 9. Open items

1. CireSnave/PM: Windows scope in practice (Radeon VII and MI50 look Linux-only; the laptop iGPU is the only measured Windows target).
2. Read ROCm's current gfx906 support and library list per OS before wave 2.
3. Install the HIP SDK on the desktop's Windows boot and record the real DLL names and the libraries it ships.
4. With the Fuel lane: how a ROCm backend maps onto `fuel-backend-contract`.
