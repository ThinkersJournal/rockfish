# `rockfish-hip` design (R5)

Design note before code. Extends `docs/DESIGN.md` s1, s2 and s5. Labels as there: **[measured]**, **[read]**, **[unverified]**. The only HIP runtime measured is the Windows laptop's `amdhip64_6.dll` (Radeon 610M, gfx1036, HIP 6.4.50501); hipRTC is absent there, so kernel launch cannot be exercised on this box without a precompiled code object.

## 1. Scope of the first PR series

| PR | Adds | Needs a GPU to test |
|---|---|---|
| R5a | `Error`, `hip()` (process-wide loaded runtime table), `runtime_version`, `device_count`, `Device` (properties, set/get) | partly |
| R5b | `Stream`, `Event`, `DeviceBuffer<T>` (alloc, free, H2D, D2H, D2D, memset, async variants) | yes |
| R5c | `Module` (load code object from bytes), `Function`, `launch` | yes, and a code object |

Each PR gets its own version from the PM. Context: HIP has no user-visible context object comparable to a CUDA driver `CUcontext`; the primary context is per device. `Device` therefore carries the ordinal and every handle remembers the device it was made on. **[read]** (HIP programming guide; not measured.)

## 2. Error type

`rockfish_hip::Error`, `#[non_exhaustive]`:

- `Load(LoadError)`: the library or one entry point is absent.
- `Api { call: &'static str, code: i32, name: String }`: a HIP call returned non-zero. `name` comes from `hipGetErrorName` when it loads, else from a built-in table of the codes in the vendored header, else `"hipError(<code>)"`.
- `Misuse(&'static str)`: a precondition the safe layer checks itself (length mismatch, zero-length alloc rejected before the driver sees it).

Every wrapper returns `Result`; none panics. `hipSuccess` is the only success value; `hipErrorNotReady` from `hipEventQuery` and `hipStreamQuery` is mapped to `Ok(false)`, not an error.

## 3. Ownership and safety

- The loaded `HipApi` is one process-wide `&'static` table returned by `hip()` (decision in section 6). Handles do not hold it; the table is never unloaded, so no handle can outlive the table that frees it.
- `DeviceBuffer<T>` (`T: Copy` plus a marker trait `DeviceRepr` for plain-old-data types, implemented for the primitive ints, `f32`, `f64`, and `f16`/`bf16` via the `half` crate if added): frees on `Drop`, ignoring the free status (logged through a hook, not a panic). Copies check lengths in the safe layer.
- `Stream` and `Event` destroy on `Drop`. `Stream::synchronize` and `Event::synchronize` are explicit.
- Raw handles are reachable through `unsafe fn as_raw`. `unsafe` code is confined to the module that calls the table; each block carries a `// SAFETY:` comment.
- Async copies take a host slice that must stay valid until the stream is synchronised; the first cut offers only the synchronous form and a pinned-host-memory type for the async one, so the safe API cannot express a use-after-free.
- Thread safety: `Device` and `Event` are `Send + Sync`; `Stream` is `Send`. **[unverified]** against HIP's documentation of per-thread current-device state: `hipSetDevice` is thread-local, so each call that depends on the device sets it first (cheap) rather than assuming it.

## 4. Kernel launch (R5c)

`Module::from_code_object(&[u8])` calls `hipModuleLoadData`; `Module::function(name)` calls `hipModuleGetFunction`; `Function::launch(grid, block, shared_mem, &stream, args)` calls `hipModuleLaunchKernel`. Arguments are a slice of `*mut c_void` built by a small `args!` macro from references to `DeviceRepr` values or `&DeviceBuffer`, which keeps the pointer-to-pointer shape HIP wants. Compiling source is `rockfish-hiprtc` (separate crate, hipRTC optional).

## 5. Tests and negative controls

- **No GPU (ordinary CI, all three OSes):** a library without HIP symbols yields `Error::Load(SymbolMissing)` from every wrapper; length-mismatch copies return `Misuse`; the error-name table is checked against the vendored header's `hipError_t_*` constants (a test parses `vendor/hip` so the table cannot drift). (Superseded: the fake-table `Drop` counting test assumed an injectable table; with the global `hip()` it was not built. Drop-once behaviour is covered by the GPU tests below.)
- **GPU (self-hosted runner, never `#[ignore]`):** alloc/copy round trip with a byte pattern, a deliberate out-of-memory request returns `Api` with name `hipErrorOutOfMemory`, an event timing pair is non-negative. On this laptop these run against `amdhip64_6.dll`; results are labelled with that device.
- A round trip that returns the buffer it was given proves nothing: the test fills the host side with a different pattern before reading back.

## 6. Open items and decisions

- **Decided: `hip()` is a process-wide, lazily loaded global** (as built; the lane's HANDOFF records the decision). Baracuda does the same: `baracuda-cuda-sys/src/driver.rs` has `pub fn driver() -> Result<&'static Driver, LoaderError>` **[read]**, and Rockfish follows that caliber and call shape.
  - **Not chosen: an explicit `Arc<Hip>` handle** (this note's earlier default, whose stated benefit was testing against a fake library).
  - **Cost accepted:** no fake-table unit tests of wrappers (section 5 note).
  - **Load failures are not cached:** a failed `hip()` retries on the next call, so installing HIP mid-process works.
  - **First use is serialised and warmed up:** `hip()` takes a process-wide lock and creates/destroys one stream on device 0 before any thread gets the table, because parallel first use hung on one machine (ROADMAP R5-H). **[measured]** on the Windows laptop only; cause unknown; not a claim that HIP is not thread-safe. Other devices and runtimes are unmeasured.
  - The same pattern is used by `rockfish-hiprtc` and `rockfish-rocblas` (`OnceLock` tables).
- Whether to depend on `half` for f16/bf16: resolved. `half` is an optional feature; Fuel hands Baracuda `half::f16`/`bf16` (half 2.7.1).
- Linux library names and soname behaviour remain **[unverified]**; no ROCm box.
