# Changelog

## 0.0.15

- New crate `rockfish-rocblas` (unpublished): safe rocBLAS layer on `rockfish-hip`. `rocblas()` process-wide table, `version()`, `Handle` (on a device's default stream, or `Handle::on_stream(&Stream)` whose lifetime ties it to the stream; host-pointer-mode scalars; `synchronize`), and `Handle::gemm` / `gemm_strided_batched` over `DeviceBuffer`s for `f32`, `f64` and, with feature `half`, `half::f16` / `bf16` (accumulating in `f32`) via `rocblas_gemm_ex`. Shapes, leading dimensions, buffer lengths, strides and device mismatches are checked before any call and reported as `Error::Misuse`; `Error::Api` names the status from `rocblas_status_to_string`, else a built-in table. Measured on one Windows laptop: `version()` returns `5.1.1.787bb67272-dirty` from Ollama's ROCm 7.1 bundle (on `PATH`), and the absent-library and missing-symbol paths give error values. NOT run on a GPU: no handle has been created against a real rocBLAS and no GEMM has run (that bundle has no gfx1036 kernels and uses `amdhip64_7.dll`, not System32's `_6`); the GEMM correctness test skips, printing why, when no rocBLAS loads. GEMM on a real device is a rig row.

## 0.0.14

- New crate `rockfish-miopen-sys` (unpublished): runtime-loaded bindings for the MIOpen HIP backend, 42 functions: handle and stream, tensor descriptors and tensor ops, convolution descriptors, workspace size, find, forward / backward data / backward weights / bias, and activation. This is the part Fuel's cuDNN-backed Conv1D/Conv2D maps onto. `hipStream_t` is the `rockfish-hip-sys` type. UNVERIFIED end to end: no MIOpen library exists on the one machine used, so neither the library names (`MIOpen.dll`, `libMIOpen.so.1`: guesses) nor the exported symbols were checked; the symbol test passes through its not-found arm and says so. The other 224 functions in `miopen.h` (pooling, batchnorm, RNN, fusion, solution API, ...) are not bound.
- `vendor/miopen`: `miopen/miopen.h` (MIOpen 3.5.1) at `ROCm/rocm-libraries` tag `rocm-7.2.4`, byte-identical to the upstream blob, with licence and manifest; `tools/verify-vendor.sh` checks it with the other sets. `miopen.h` is C++-valid but not C-valid, so `tools/gen-bindings miopen` parses it as C++.

## 0.0.13

- New crate `rockfish-hipblaslt-sys` (unpublished): runtime-loaded bindings for the whole hipBLASLt C API in `hipblaslt.h`, 24 functions (handle, matrix layout, matmul descriptor and preference, algorithm heuristic, `hipblasLtMatmul`, matrix transform). `hipStream_t` is the `rockfish-hip-sys` type. Measured on one Windows laptop against Ollama's ROCm 7.1 `libhipblaslt.dll`: it exports exactly these 24 and all resolve (headers are hipBLASLt 1.2.2). NOT run on a GPU: no handle has been created. Linux sonames UNVERIFIED. The C++ extension API (`hipblaslt-ext.hpp`) is not bound.
- `vendor/hipblaslt`: 4 hipBLASLt headers and `hipblas-common.h` (from `projects/hipblas-common`) at `ROCm/rocm-libraries` tag `rocm-7.2.4`; `vendor/hip` gains `hip/library_types.h` (`hipDataType`) from `rocm-systems` at the pinned commit. All byte-identical to upstream blobs; `tools/verify-vendor.sh` checks all three sets.
- `tools/gen-bindings` takes a `hipblaslt` target (header parsed as C++, HIP and hipBLAS-side stand-ins for headers that are not vendored).

## 0.0.12

- New crate `rockfish-rocblas-sys` (unpublished): runtime-loaded rocBLAS bindings, 56 functions (handle, stream, pointer/atomics/math mode, version, vector/matrix copies, level 1 `s`/`d` scal axpy dot nrm2 asum copy swap iamax iamin, gemv, geam, and the GEMM family: `h`/`s`/`d` gemm with batched and strided-batched forms, `gemm_ex` and its batched forms). `hipStream_t` is the `rockfish-hip-sys` type. Measured on one Windows laptop against Ollama's ROCm 7.1 `rocblas.dll` (rocBLAS 5.1.1, headers are 5.2.0): all 56 symbols resolve and `rocblas_get_version_string` answers. NOT run on a GPU: no call that touches a device has been made, and that bundle ships no gfx1036 kernels (this laptop's iGPU). Linux sonames UNVERIFIED.
- `vendor/rocblas`: the 10 rocBLAS headers at `ROCm/rocm-libraries` tag `rocm-7.2.4`, byte-identical to the upstream blobs with licence and manifest; `tools/verify-vendor.sh` now checks both vendored sets (`--online` also compares against the forge).
- `tools/gen-bindings` takes a `rocblas` target (own include dir, stubs for CMake-generated headers, force-included HIP header, types reused from another `-sys` crate). Every generated table gains `unresolved()`, the bound names the loaded library does not export; the HIP and hipRTC output changed only by that addition.

## 0.0.11

- `rockfish-hip`: `PinnedBuffer<T>` (`hipHostMalloc`, derefs to `[T]`, freed on drop) and asynchronous copies `DeviceBuffer::copy_from_pinned_async` / `copy_to_pinned_async` (`hipMemcpyAsync`). Each returns a `PendingCopy` guard that borrows the buffers and the stream until `wait()` or drop synchronises; `mem::forget` on the guard is a documented hole. Measured on the Radeon 610M: a 4096-element round trip through pinned memory, including drop-as-wait, plus a refused cross-device stream; one device only, so cross-device behaviour is UNVERIFIED.

## 0.0.10

- New crate `rockfish-hiprtc` (unpublished): optional safe hipRTC wrapper. `hiprtc()` process-wide table, `Program` (new, name expressions, compile with options, log, code object, lowered names; destroyed on drop), `version()`, `Error` with a `Compile { log }` variant. UNVERIFIED end to end: the laptop has no hipRTC (`hiprtc.dll` not found, measured), so the compile path has never run; tested are the absent-library error, a missing entry point, NUL-byte refusal and log handling.

## 0.0.9

- `rockfish-hip`: `Module::from_code_object` (`hipModuleLoadData`), `Function` (borrows its module), `LaunchConfig`, `KernelArgs` (values and device buffers, borrowed until the call returns) and `unsafe Function::launch` (`hipModuleLaunchKernel`). Measured on the Radeon 610M: a garbage image is rejected with a named error. NOT measured: a successful kernel launch, because no GPU compiler is available on the laptop; that test runs when `ROCKFISH_TEST_CODE_OBJECT` names a code object and otherwise says it did not run.

## 0.0.8

- `rockfish-hip`: `Stream`, `Event` (record, synchronize, poll, elapsed), `DeviceBuffer<T: DeviceRepr>` (alloc, host/device copies, `fill_zero`, freed on drop), `DeviceRepr` for the primitive ints and floats. Optional feature `half` implements `DeviceRepr` for `half::f16`/`bf16` (`half` 2.7.1, as Fuel resolves). Synchronous copies only; async copies come later. `hip()` now warms the runtime up under a lock (see its docs for the measured hang it prevents). GPU assertions UNVERIFIED in CI; measured locally on the Radeon 610M.

## 0.0.7

- New crate `rockfish-hip` (unpublished), first slice: `Error` (load / API-with-name / misuse), process-wide `hip()` table like Baracuda's `driver()`, `runtime_version`, `device_count`, `Device` (name, memory, set current). GPU assertions are UNVERIFIED in CI; run locally on the Radeon 610M.
- `rockfish-hip-sys` now exports `ERROR_NAMES` (status code to name), emitted by `tools/gen-bindings`; fixed `rockfish-hip-sys` test imports that failed to compile on macOS.

## 0.0.6

- `docs/DESIGN-R5.md`: design of the safe `rockfish-hip` layer (error type, ownership, launch, tests). No code.

## 0.0.5

- New crates `rockfish-hip-sys` and `rockfish-hiprtc-sys` (unpublished): runtime-loaded HIP and hipRTC bindings generated by `tools/gen-bindings` from the vendored headers; a missing library or entry point is an error value. Measured on a Windows laptop: loads the driver-bundled `amdhip64_6.dll` and reports the Radeon 610M.

## 0.0.4

- Vendored the HIP public headers needed to generate bindings (`vendor/hip/`, from ROCm/rocm-systems rocm-7.2.4), each verified byte-for-byte against its upstream git blob; `THIRD-PARTY-NOTICES.md`, `tools/verify-vendor.sh`, CI job `vendor`.

## 0.0.3

- New crate `rockfish-core` (unpublished): `LibrarySpec` per-OS names, search order (`ROCKFISH_<NAME>_PATH`, `HIP_PATH`, `ROCM_PATH`, loader), `Library::open` and `Library::symbol` returning `LoadError` instead of panicking.

## 0.0.2

- Docs: `docs/DESIGN.md` (R3) and `docs/SPIKE-R3b-versions.md` (HIP binding differences across releases).

## 0.0.1

- Docs: `docs/SURVEY.md` (R2), the survey of existing ROCm/HIP crates with a Windows HIP runtime measurement. Skeleton crates stay unpublished (`publish = false`).
