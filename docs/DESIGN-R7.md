# HIP C++ parser and emitter (R7)

Design note before code. Labels as in `docs/DESIGN.md`: **[measured]**, **[read]**, **[unverified]**.

**Assignment, verbatim (CireSnave 2026-10-30, as relayed by the PM and recorded in Unpopped's `docs/parser-emitter-ownership-map.md` section 9):** *"HIP C++ should be owned by Rockfish and Rockfish should write the parser and emitter for that."* Unpopped holds neither for any target; it keeps the IR, the traits and one reference emitter (`unpopped-cpu-c`). **[read]**

All Unpopped facts below are read from `origin/main` of Unpopped (`Cargo.toml` version 0.37.0 when read), not from a working tree.

## 1. The contract Rockfish implements

**Emit side [read]:** `unpopped::backend::Backend` (`crates/unpopped/src/backend.rs`).

- `name() -> &str`: target language, `"hip"` here.
- `provider() -> &str`: who supplies the backend, emitted verbatim into the FKC `kernel_source:` field; required, no default. Baracuda's returns `"baracuda"`; ours returns `"rockfish"`.
- `lower(&KernelPlan) -> Result<GeneratedKernel, LowerError>` and the defaulted `lower_kernel(&KernelFn)` (default: typed refusal `LowerError::UnsupportedKernel`).
- `supports_dtype(dtype, target) -> DtypeSupport` (`Supported` / `Unsupported` / `NotExpressible`, `#[non_exhaustive]`). When unsure, answer `Unsupported`.
- **Refuse rather than approximate.** A backend that cannot spell something exactly returns `Err`; it never falls through to code that compiles but computes something else.
- `GeneratedKernel::new(name, source)` is the only constructor from outside the crate (`#[non_exhaustive]`; provenance is stamped by `unpopped::generate`).

**Parse side [read]:** `unpopped::convert::Frontend` (feature `convert`, off by default), a `#[non_exhaustive]` descriptor built with `Frontend::new(name, kernel_marker, residue, out_name, in_prefix, parse)` and tuned with `with_intrinsics`. Only `parse: fn(&str) -> Option<Tree>` needs a grammar crate. The shared walkers read C-family node and field names (notably a `subscript_expression`'s `indices` field), so the grammar must be `tree-sitter-cpp`-shaped. Math-call tables default to CUDA's spellings (`expf`, `sqrtf`); a fast-math variant (`__expf`) must map to `None`, not to the accurate op.

**Compile side [read]:** `unpopped::jit::Compiler` (`compile(source, entry, max_compile_ms) -> Result<Vec<u8>, String>`, plus `artifact_kind()` defaulting to `ArtifactKind::Ptx`).

**Precedent [read]:** Baracuda splits the same way: `baracuda-cuda-emit` (the `Cuda` backend, `cuda.rs` 884 KB, NVRTC compiler behind a feature) and `baracuda-cuda-parse` (`lib.rs` 3.7 KB: `parse_cuda`, the residue list, the `CUDA` descriptor; deps `unpopped` with `convert`, `tree-sitter` 0.27, `tree-sitter-cuda` 0.21).

## 2. Crates

| Crate | Contents | Depends on |
|---|---|---|
| `rockfish-hip-parse` | `parse_hip`, `HIP_RESIDUE`, the `HIP` `Frontend` | `unpopped` (feature `convert`), `tree-sitter`, a grammar crate (section 4) |
| `rockfish-hip-emit` | `Hip` `Backend` | `unpopped`, `unpopped-vocab` |
| `rockfish-hip-emit` feature `hiprtc` | `HipRtcCompiler: unpopped::jit::Compiler` over `rockfish-hiprtc` | `rockfish-hiprtc` |

Both stay out of `rockfish-hip`: neither needs a GPU or a HIP runtime to build or test, and the safe runtime layer should not pull in Unpopped. Dependencies are crates.io versions only (CireSnave 2026-10-07), so `unpopped` is taken from the registry at its latest published version, never a path.

**Not reused from Baracuda:** `baracuda-cuda-emit` carries its own closed-set copy of the scalar spellers (`cfamily_shadow`) so as not to import from `unpopped::cfamily`; Rockfish follows that, and does not depend on a Baracuda crate. Reading its source for structure is fine; copying text needs licence and credit kept (CireSnave 2026-09-17, "I am not a plageurist"), so any copied helper is traced and credited in the file.

## 3. Emitter: what differs from CUDA

HIP C++ is close to CUDA C++ at the source level, which is the temptation and the risk. The design rule is: **every spelling is stated by the emitter, none inherited by renaming CUDA output.** A textual CUDA-to-HIP rewrite of `Cuda` output would couple Rockfish to Baracuda's 884 KB file and to its changes, and would silently carry any CUDA-only construct across.

Items the emitter must decide, each **[unverified]** until checked against a HIP compiler or the headers (the vendored `vendor/hip/include/hip` has `hip_runtime_api.h`, `hiprtc.h` and the `amd_detail` dir, but not `hip_runtime.h`, `hip_fp16.h` or `hip_bf16.h` **[measured]**: listing the directory):

- Includes: `hip/hip_runtime.h`, and the half/bfloat16 headers for those dtypes.
- Kernel entry: `extern "C" __global__ void name(...)`, as the CUDA backend does; `__launch_bounds__` use.
- Thread/block indices (`threadIdx.x` form, which HIP accepts alongside its older `hipThreadIdx_x` macros) **[unverified]**.
- f16 and bf16 type names and conversion intrinsics.
- **Wavefront size is not 32.** `hipDeviceProp_t.warpSize` exists **[measured]** in the vendored header; AMD's gfx906 (the rig's GPU) is a 64-wide wavefront part **[read, AMD documentation; not measured]**, while this laptop's gfx1036 is RDNA. Any warp-level reduction, shuffle or ballot must take the width from the target, never assume 32. Whether `KernelPlan` carries a subgroup size is not yet read; that is the first thing R7b reads.
- Atomics, shared memory and barrier spellings: not needed for the first slice (section 5).
- Fast-math: the same rule as the parse side; never lower a fast variant as the accurate op or the reverse.

`supports_dtype` answers per target (a `gfx` architecture string). Until a dtype's availability on an architecture is measured or read from AMD's documentation, the answer is `Unsupported`, which declines safely.

## 4. Parser: grammar choice

No HIP tree-sitter grammar is on crates.io **[measured]**: `cargo search tree-sitter-hip` returned nothing; `tree-sitter-cuda` 0.21.2 and `tree-sitter-cpp` 0.23.4 are there. Options:

1. **`tree-sitter-cuda`** (what Baracuda uses). HIP kernels use the same `__global__`, `__shared__`, `<<<...>>>` and index syntax, so it is the likely fit. **[unverified]** for HIP-specific syntax (`hipLaunchKernelGGL(...)`, `__hip_*` attributes).
2. **`tree-sitter-cpp`**. `Frontend` requires a cpp-shaped tree and the parse is error-tolerant, so unknown attributes degrade to ERROR nodes rather than failing. Lower coverage of CUDA-style syntax.
3. Write a grammar: out of scope; a last resort.

**Decision for R7a:** build with `tree-sitter-cuda`, run a corpus of HIP kernels through it, and record which parse cleanly; fall back to `tree-sitter-cpp` for constructs that do not, and state it. The corpus is hand-written kernels in our own tests, not copied from AMD or third-party samples without credit.

Residue (constructs the neutral IR does not hold, refused rather than lifted): shared memory, atomics, barriers, library calls (`hipblas*`, `rocblas_*`, `__shfl*`, wave intrinsics), inline asm, `hipLaunchKernelGGL` host code. The list is data in `HIP_RESIDUE`, tested by one positive and one negative case per entry.

## 5. Staging (each its own PR; versions allocated by the PM)

| PR | Adds | Verifiable here? |
|---|---|---|
| R7a | `rockfish-hip-parse`: grammar choice settled by corpus, `HIP` frontend, elementwise lift | yes (no GPU) |
| R7b | `rockfish-hip-emit`: `Hip` backend, elementwise `lower`; `supports_dtype` | emitted text yes; compilation **no** (no amdgcn compiler on this box **[measured]**: HANDOFF) |
| R7c | reductions and scans, matching what the parse side lifts | text only |
| R7d | `HipRtcCompiler` over `rockfish-hiprtc` | **no**: no `hiprtc.dll` here **[measured]**; rig row |

## 6. Tests and negative controls

- **Round trip:** emit with `Hip`, parse the result with `HIP`, and require the lifted IR to equal the original. A control: change one operator in the emitted text and require the round trip to differ. (A round trip that returns what it was given proves nothing, as in DESIGN-R5.)
- **Refusal:** each residue entry, an unsupported dtype and a plan shape outside scope produce a typed error, never source.
- **Golden text** for emitted kernels, reviewed by reading, not regenerated blindly.
- **No-claim rule:** nothing emitted is called "compiles" or "runs" until it has been through a HIP compiler. Until the rig exists the docs and CHANGELOG say "emits text; compilation UNVERIFIED".
- **Rig rows (wait for the dual-boot window):** compile the golden kernels with `hipcc`/hipRTC for gfx906 (Linux boot) and launch one through `rockfish-hip`; this is also the launch test already open in HANDOFF.

## 7. Rulings and requests to other lanes

- **Version numbering (PM ruling 2026-10-31, from CireSnave's one-version rule):** `rockfish-hip-emit` and `rockfish-hip-parse` take the **track-Unpopped exception exactly like `baracuda-cuda-emit`**. CireSnave's rule: a crate designed to work with another project keeps the version of the project it works with, so the number answers "which Unpopped does this work with?" These two crates therefore do not follow Rockfish's workspace number; each crate's `Cargo.toml` carries a comment saying so. The PM allocates the number at gate time. (Baracuda's own `baracuda-cuda-parse` comment documents MAJOR.MINOR following Unpopped with PATCH as its own counter **[read]**; the exact scheme here is the PM's to allocate.)
- **Unpopped, `ArtifactKind` (request, via the PM):** `unpopped::jit::ArtifactKind` has `Ptx`, `Cubin` and `Stub` **[read]**, is **not** `#[non_exhaustive]`, and `Compiler::artifact_kind()` defaults to `Ptx`. `HipRtcCompiler` (R7d) would otherwise report a HIP code object as PTX. Proposal: add a variant `HipCodeObject` (an AMD GPU code object, architecture-specific like `Cubin`) and mark the enum `#[non_exhaustive]` so the next addition is not breaking. **The `#[non_exhaustive]` marking is itself breaking for any out-of-crate exhaustive `match`, and so is adding a variant to the current enum, so it needs Unpopped's next breaking release: pre-1.0 that is 0.38.0 (current 0.37.0 when read), not a 0.37.x patch.** I need that release before R7d only; R7a-R7c do not touch `ArtifactKind`.
- **Fuel (R8):** whether Fuel will consume the `Hip` backend through Unpopped's `generate`/`synthesize` path as it does Baracuda's. Not read yet.
- Which crate names are reserved on crates.io for the emit/parse crates is **[unverified]** here.
