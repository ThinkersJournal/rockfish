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
- **Wavefront size is not 32.** `hipDeviceProp_t.warpSize` exists **[measured]** in the vendored header; AMD's gfx906 (the rig's GPU) is a 64-wide wavefront part **[read, AMD documentation; not measured]**, while this laptop's gfx1036 is RDNA. Any warp-level reduction, shuffle or ballot must take the width from the target, never assume 32. **`KernelPlan` carries no subgroup / wavefront size [read]** (Unpopped 0.38.4 `plan.rs`: the only `warp` mentions are doc comments on the scan and reduce schedules; `grep` for `subgroup` finds none). Elementwise needs none. R7c must take the width from the target or from the device-side `warpSize`, and must not assume 32.
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
| R7b | `rockfish-hip-emit`: `Hip` backend, elementwise `lower`; `supports_dtype` (see section 5a for what shipped) | emitted text yes; compilation **no** (no amdgcn compiler on this box **[measured]**: HANDOFF) |
| R7c | reductions and scans, matching what the parse side lifts | text only |
| R7d | `HipRtcCompiler` over `rockfish-hiprtc` | **no**: no `hiprtc.dll` here **[measured]**; rig row |

## 5a. What R7b shipped, and what it decided

Scope is one cell: `Schedule::Scalar`, one output, `f32`, target namespace `rocm:` (Unpopped's vectors spell an AMD target `rocm:gfx942`). Everything else is a typed `LowerError`, never source. Read from Unpopped 0.38.4 **[read]**; the emitted text has not been through a HIP compiler.

- **Closed spelling set, stated by the emitter.** Infix `+ - * /`; unary `Neg`, `fabsf`, `sqrtf`, `expf`, `logf`, `tanhf`, `sinf`, `cosf`, `floorf`, `ceilf`, `truncf`, `exp2f`; binary `fmaxf`, `fminf` (IEEE, NaN-suppressing: the IR's `FmaxIeee`/`FminIeee`), `powf`, `atan2f`, `copysignf`. This is exactly the inverse of the call names the shared lifter reads, so the round trip is a real test. The NaN-propagating `Max`/`Min`, `Fma`, `Select`, comparisons, remainders, `Rsqrt`/`Recip`/`Sqr`, `Round` (ties-to-even vs C `roundf`) and the rest are declined until a spelling is decided and checked.
- **Floating-point contraction.** The kernel body starts with `#pragma clang fp contract(off)`. Clang's default for HIP is, as far as the author has read, fused contraction, which would turn the IR's two-rounding `Add(Mul(a, b), c)` into one rounding. **[unverified]**: the pragma's effect is not checked against a HIP compiler; if it is wrong the kernel is still a valid-looking kernel computing a different function, which is why it is stated and tested for in the golden.
- **Constants.** At an `f32` compute dtype the IR rounds a constant to `f32`, round to nearest even (`unpopped::optimize::Compute::F32`), and `generate` does not run the optimizer, so the emitter spells `(v as f32)` once, with an `f` suffix and the shortest digits that name that `f32`. Non-finite values, overflow to infinity and underflow of a nonzero value to zero are refused. The HIP frontend reads a literal's digits as an `f64`, so the lifted constant equals the authored one after the same `f32` rounding, not bit for bit (tested that way).
- **No `__restrict__`.** It would assert that `out` does not alias an input, which the emitter has no basis to claim.
- **Grid-stride loop** with the first index widened to `long long` before the multiply, so the result does not depend on the launch covering `n` exactly.
- **`supports_dtype`:** `Supported` only for `f32` on a `rocm:` target (a fact about the language and this emitter's scope, not about any gfx part's instructions). Every other dtype, and every other namespace, is `Unsupported`.
- **No status text in the emitted source.** A comment saying "unverified" would be false the day the rig verifies it.
- **Not done in R7b:** `Vectorized` and `Strided` schedules (an aligned contiguous cell keys `Vectorized`, so a default `rocm:` cell is declined today; the tests use a small alignment to get `Scalar`), runtime scalar parameters, multiple outputs, `f16`/`bf16`/`f64`/integers.

## 5b. What R7c shipped, and what it decided

Two more schedules, same `f32` / `rocm:` / one-output scope, same refusal rule. Read from Unpopped 0.38.4 **[read]**; not compiled by a HIP compiler.

- **Sequential reduction** (`Schedule::Reduction`, class `InnerContig`): `Sum`, `Mean`, `Prod`, `Max`, `Min` over the contiguous trailing axis, identity epilogue only. One thread per output row; a block-parallel reduction is a later slice. Other classes (`Outer`, `Middle`, `Multi`), a fused post-expression and a non-identity `ReduceOp` are typed refusals.
- **Serial scan** (`Schedule::Scan`, `block: false`, the deterministic base): `Sum`, `Prod`, `Max`, `Min`, `reverse` and `exclusive` both, innermost axis only, identity epilogue. The cooperative block scan, and `Mean` (not a monoid), are refused.
- **Wavefront width: none is used, and none is assumed.** Neither slice has a cross-lane step. `KernelPlan` carries no width and, per the PM (Unpopped `4ec1a562`, 2026-10-31), will not: plans stay target-neutral and the emitter owns the width. The first emitter that needs one (block reduction, block scan) must read it from the target or device-side `warpSize`, never 32; a test pins that these two slices contain no `warpSize`, `__shfl`, barrier or shared memory.
- **Accumulator is `double`**, narrowed to `float` at the store, because the oracle (`eval_reduction`, `eval_scan`) folds in `f64`. This is the emitter's reading of `ReductionAccum::Auto`. Max/Min follow the oracle: a NaN element wins and a NaN accumulator stays; an empty axis gives `0` (Mean gives `0/0 = NaN`). Exclusive scans store the identity (`0`, `1`, `-inf`, `+inf`) first.
- **Launch ABI** is `(in0.., out, rows, k)` over contiguous rows, one thread per row, grid-stride. The plan carries no extents, so this is a stated contract, not a checked one.
- **Measured (one-off, not in CI):** the 5 reduction and 16 scan kernels were compiled with the Visual Studio `clang++` against a shim for the HIP built-ins and run on the host over 3 rows of 4 (one with a NaN) and `k = 0`; the outputs matched a Python transcription of the oracle's loops. **That reference shares the author and the algorithm with the emitter**, so it checks the C++ spelling, not the semantics; it says nothing about a HIP compiler, `double` on a gfx part, or the `fp contract` pragma.
- **Not round-trippable:** the HIP frontend lifts `acc += in0[i]` loops; the emitted `double acc ... acc = acc + v` form is not claimed to lift, and no test says it does.
- **Not done:** block-parallel reduce and scan (needs the width), `RowReduce`, `Window`, `Contraction`, `RowSort`, multi-axis reductions, integer and half dtypes.

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
