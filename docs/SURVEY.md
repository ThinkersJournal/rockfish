# Rockfish survey (R2)

Options first, recommendation last, no crate code. Measured 2026-10-30 unless a line says otherwise. Three kinds of claim appear below and are labelled: **[measured]** (I ran it, ref given), **[read]** (read from published source or the forge, nothing built), **[unverified]** (stated by someone or remembered; check before building on it). There is no ROCm test box yet, so nothing here has run on a discrete AMD card.

## 1. Existing Rust crates [read]

Sources: crates.io API and the published `.crate` tarballs, unpacked and read on 2026-10-30. Download counts as of that day.

| Crate | Version | Licence | Downloads | Scope | Loading | Windows |
|---|---|---|---|---|---|---|
| `cubecl-hip-sys` | 7.14.6085001 (2026-09-15) | MIT OR Apache-2.0 | 1,927,250 | bindgen for HIP runtime + hipRTC, one binding file per HIP build (`bindings_41134.rs` ... `bindings_60850.rs`) | runtime, `libloading`; `is_available()`; a missing library or symbol **panics** inside the wrapper | README says Linux only; `dynamic.rs` has a `.dll` branch (`amdhip64.dll`, `hiprtc.dll`), unmeasured, and the only test is `cfg(target_os = "linux")` |
| `rocm-rs` | 0.5.2 (2026-07-03) | MIT | 5,852 | raw bindings: HIP, rocBLAS, MIOpen, rocFFT, rocRAND, rocSOLVER, rocSPARSE, rocprofiler, rocm-smi; headers vendored in `include/` | **link time** (`build.rs` emits `rustc-link-lib`) | no sign of support |
| `rocmrc` | 0.5.1 (2026-08-28) | MIT OR Apache-2.0 | 401 | cudarc-style: HIP, hipRTC, rocBLAS, hipBLASLt; per-ROCm-version bindings behind `rocm-0NNNN` features | optional `dynamic-loading`; candidate names built from `DLL_PREFIX`/`DLL_SUFFIX` and a default of `/opt/rocm` | some `cfg(windows)` for external memory; library names are POSIX-shaped |
| `rocm-sys` | 0.4.1 (2026-10-12) | MIT | 147 | HIP bindings keyed on ROCm version | static or `dynamic-loading` | not examined |
| `hanzo-rocm` | not examined | | | PM said it looks like a copy of `rocm-rs` | | skip |

What this says:

- Nobody covers the HIP-runtime-through-math-library stack **with runtime loading on both OSes**. The popular crate (`cubecl-hip-sys`) covers two libraries; the wide crate (`rocm-rs`) links at build time.
- `cubecl-hip-sys` already solved the hard part of **many HIP versions**: it generates a binding file per HIP build and `build.rs` picks one from `hipconfig`, rewriting bindgen output into wrappers that call `dynamic::load`. Its two weaknesses for us: it panics on a missing library or symbol (we want a `Result` so a lane can degrade), and the version is chosen at **build** time from the build machine's `hipconfig`, which is wrong for a binary built on one machine and run on another.
- Licences are all compatible with ours (MIT OR Apache-2.0). Nothing has been copied; anything we copy keeps its licence and credit.

## 2. What loads on Windows [measured]

Checked on the laptop (Windows 11, `AMD Radeon(TM) 610M` iGPU, driver 32.0.21045.16010, plus an RTX 4070 Laptop GPU). No HIP SDK installed (`HIP_PATH`/`ROCM_PATH` unset, no `hipconfig`, no `C:\Program Files\AMD\ROCm`).

- `C:\Windows\System32\amdhip64_6.dll` **exists**, 17,906,704 bytes, product string `amdhip64 6.4 AMD-APP.dbg (3652.0)`. It arrived with the Adrenalin driver, not an SDK.
- Loaded with Python `ctypes.WinDLL` (throwaway script, not committed): `hipGetDeviceCount` returns 0 (success) with count 1; `hipRuntimeGetVersion` = 60450101; `hipDeviceGetName` = `AMD Radeon(TM) 610M`; the architecture string in `hipGetDevicePropertiesR0600` ends `gfx1036`; `hipMalloc` of 1 MiB succeeds and a 16-byte host-to-device then device-to-host `hipMemcpy` round-trips equal; `hipMemGetInfo` reports 36,460 of 36,769 MiB free (shared system memory, as expected for an iGPU).
- Exports present: `hipMalloc`, `hipModuleLoadData`, `hipGraphLaunch`. **Not** exported: `hiprtcCreateProgram` / `hipRtcCreateProgram`. **No hipRTC DLL in System32**; the only AMD neighbours are `amd_comgr_2.dll` and `hiprt0200064.dll` (HIP RT, ray tracing, a different library).
- So: **the HIP runtime is reachable on stock Windows with only a graphics driver, and the DLL is `amdhip64_6.dll`, not the `amdhip64.dll` that `cubecl-hip-sys` looks for.** A loader must try versioned names (`amdhip64_7.dll`, `amdhip64_6.dll`, `amdhip64.dll`) and honour `HIP_PATH`.
- Not measured: compiling and launching a kernel (needs hipRTC or a precompiled code object, neither present here); hipRTC, rocBLAS, MIOpen and the other libraries' Windows names and availability (HIP SDK for Windows, **[unverified]**: I have not read AMD's current table of libraries shipped on Windows); any discrete card; Linux at all.

## 3. Options

**A. Depend on `cubecl-hip-sys` for the HIP layer, generate our own for the math libraries.** Cheapest start, shares fixes with the largest user base. Cost: panic-on-missing, build-time version pick, Windows names wrong or unproven, release cadence tied to HIP patch numbers. Our safe layer would sit on a `-sys` whose error model we cannot change.

**B. Generate everything ourselves** (runtime loading with a `Result`-returning symbol table, one `-sys` crate per library like Baracuda). Most work, full control of loading, per-OS names in one table, version handling at run time, same crate shape as Baracuda so Fuel consumes it the same way. Cost: bindgen over HIP headers per version, and keeping up with ROCm releases ourselves.

**C. Build on `rocmrc`.** Closest in spirit (cudarc-style, optional dynamic loading) but 401 downloads, one maintainer, POSIX-shaped names, build-time version features.

**D. Build on `rocm-rs`.** Widest library coverage but link-time and unsafe-by-default; wrong shape for runtime loading.

## 4. Recommendation

**B, borrowing A's per-version-bindings idea (and crediting it if any code is copied).** For R3 to confirm:

1. One `-sys` crate per library, same split as Baracuda (`rockfish-hip-sys`, `rockfish-rocblas-sys`, ...), with `rockfish-sys` as the umbrella that already exists on crates.io. Runtime-loaded; a missing library is `Err`, a missing symbol is `Err` for that function only.
2. A single per-OS library-name table (`amdhip64_7.dll`, `amdhip64_6.dll`, `amdhip64.dll`, `libamdhip64.so[.N]`, ...), searched via `HIP_PATH`/`ROCM_PATH`, then the default loader path. Windows names are verified against a real HIP SDK before they are written down.
3. HIP version chosen **at run time** from `hipRuntimeGetVersion` where bindings differ, not from the build machine.
4. Safe layer cudarc-style on top; Windows and Linux in CI where a runner allows; hardware tests via the self-hosted-runner rule once a ROCm box exists.
5. Math libraries in order of value to Fuel: rocBLAS/hipBLASLt first, then rocRAND, MIOpen, rocFFT, rocSOLVER, rocSPARSE.
6. HIP C++ parser and emitter (R7) as their own crates, Baracuda's `cuda-parse`/`cuda-emit` being the precedent, on Unpopped's `Frontend`/`Backend` contract.

CireSnave has not ruled on A versus B. The PM's earlier plan leaned toward A for the runtime layer; the Windows measurement is the main evidence against: the thing A gives for free does not find the library on the one machine I could test.

## 5. Open questions for R3

- Which HIP SDK for Windows version to target, and which of rocBLAS/hipBLASLt/MIOpen/rocFFT/rocRAND/rocSOLVER/rocSPARSE ship on Windows (read AMD's current table, then install the SDK on the desktop rig's Windows boot).
- Whether the driver-bundled `amdhip64_6.dll` is a first-class case (no SDK needed, but no hipRTC), which decides whether Windows kernels come from hipRTC or ahead-of-time code objects.
- The laptop's `gfx1036` iGPU is a development target for the runtime (allocation, copies), not for benchmarks.
- What Fuel needs from a backend, from the Fuel lane, before the safe API is shaped.
