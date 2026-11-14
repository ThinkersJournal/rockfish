# Changelog

## 0.0.40

- `rockfish-hip-emit` spells the strided elementwise cell at `i32`/`i64`/`u32`/`u64` (the float-only refusal and the schedule gate for integers are lifted). The body goes through the existing integer seams (wrapping `Add`/`Sub`/`Mul`, total `Div`/shifts, comparisons, `Select`); addressing, unravel and the launch ABI are the float cell's, unchanged. Still refused: `Coord` at an integer dtype (Unpopped's plan gate refuses it: "int-literal coordinate spelling is a follow-up"), `f16`/`bf16` on this schedule, strided window, and strided reduction/scan (not built). Tests: `tests/strided.rs`, text only. **UNVERIFIED against a HIP compiler or an executed run (rig row); no host differential run.** Workspace 0.0.40; `rockfish-hip-emit` and `rockfish-hip-parse` keep tracking Unpopped 0.45.12.

## 0.0.39

- `rockfish-hip-emit` spells the strided elementwise cell (`Schedule::Strided`, `Access::Elementwise`) at `f32`/`f64`: a grid-stride loop over the `n` output elements, each unravelled row-major into `c0..c{rank-1}` against the iteration shape, every input and the output addressed by its own element strides (a transposed or sliced operand is read in place; a broadcast axis drops its term; an operand broadcast on every axis is loaded once before the loop). A `Coord(d)` leaf now lowers here (`((float)c{d})`); on the contiguous cells it is still a typed decline. Launch ABI (stated, not checked; element strides): `(in0.., out, shape0.., s{i}_{d}.., so_0.., n, params..)`, the argument names of `baracuda-cuda-emit`'s strided kernel. Still refused, typed: views, gather, scatter and base offsets (`check_plan_shape`), integers and `f16`/`bf16` on this schedule. Tests: `tests/strided.rs`, text only. **UNVERIFIED against a HIP compiler or an executed run (rig row); no host differential run this time.** Workspace 0.0.39; `rockfish-hip-emit` and `rockfish-hip-parse` keep tracking Unpopped 0.45.12.

## 0.0.38

- `rockfish-hip-emit` spells the sliding-window pool (`Schedule::Window`, the pooling family) at `f32`/`f64`: `Sum`, `Mean`, `Prod`, `Max`, `Min`, over any pooled axis. One thread per output cell (grid-stride), an independent fixed-order fold over taps `p = o*stride - pad_lo + kk*dilation` (skipped outside `[0, k_in)`), `double` accumulator narrowed once at the store, as Unpopped's `eval_window` folds in `f64` (0.45.12). `Mean` divides by `size` (`count_include_pad`) or the valid-tap count (`0` when none); `Max`/`Min` keep the oracle's NaN rule and answer the monoid identity when no tap was valid. The pre-map is lowered over each tap by the existing seams. Launch ABI (stated, not checked): `(in0.., out, outer, k_in, k_out, inner, params..)` with the tensor split `(outer, k, inner)` around the pooled axis; the caller owes the oracle's every-window-has-a-real-tap precondition. Refused, typed: integer dtypes, `f16`/`bf16`, a fused post-expression, a non-dense (broadcast, flipped, dequantised) input. Strided and block-parallel schedules are unchanged (not built; block-parallel is a ruling, see `docs/PATENT-SWEEP.md`). Tests: `tests/window.rs`, text only. **UNVERIFIED against a HIP compiler or an executed run (rig row); no host differential run this time.** Workspace 0.0.38; `rockfish-hip-emit` and `rockfish-hip-parse` keep tracking Unpopped 0.45.12.

## 0.0.37

- `rockfish-hip-emit` spells the serial prefix scan at `i32`/`i64`/`u32`/`u64`: `Sum`, `Prod`, `Max`, `Min`, forward or reverse, inclusive or exclusive (the plan already admitted them; the emitter refused any integer scan). `Sum`/`Prod` accumulate in the unsigned type of the same width, so signed overflow is never undefined. `Max`/`Min` compare in the dtype with a `have` flag; an exclusive `Max`/`Min` scan writes the dtype extreme (`Max` the minimum, `Min` the maximum) before the first element is folded, as the oracle (`eval_scan` / `scan_identity`, Unpopped 0.45.12) does. Integer `Mean` and the bitwise folds emit nothing (the plan gate and the oracle refuse them). A non-identity `post` is still refused, as for floats. Tests: `tests/int_scan.rs`, text only. UNVERIFIED against a HIP compiler or an executed run (rig row); no host differential run this time. Workspace 0.0.37; `rockfish-hip-emit` and `rockfish-hip-parse` keep tracking Unpopped 0.45.12.

## 0.0.36

- `rockfish-hip-emit` spells the sequential last-axis reduction at `i32`/`i64`/`u32`/`u64`: `Sum`, `Prod`, `Max`, `Min` (the plan already admitted them; the emitter refused any integer off the scalar cell). `Sum`/`Prod` accumulate in the unsigned type of the same width (wrapping is defined, and wrapping at the end equals wrapping at every step, which is Unpopped's oracle rule), stored as `(T)acc`. `Max`/`Min` compare in the dtype (no NaN rule) and an empty axis answers `0`, as the oracle does. The body goes through the integer seams (a total `Div`, `Select` or comparison inside works) and its helpers are emitted ahead of the kernel. Refused, typed: integer `Mean`, `And`/`Or`/`Xor`, and an integer scan. Still refused by Unpopped's own plan gate, not here: integer elementwise `Max`/`Min`. Tests: `tests/int_fold.rs`. **Text only: no HIP compiler has seen it (rig row).** One-off host check, not in CI: the emitted kernels for all 16 dtype/op cells, through a shim `hip_runtime.h`, compiled with host clang++ at `-O0` and `-O2` and run over boundary-heavy random inputs at `k` in {0,1,2,3,5,17,64} against a `__int128` reference with wrap at every step: 420,000 cases per cell and level, 0 mismatches in all 32 runs. A seeded mutant (i32 `Max` without the `have` guard) was caught: 32,004 mismatches.

## 0.0.35

- `rockfish-hip-emit` spells `f16` and `bf16` on a `rocm:` target (`Hip::supports_dtype` answers `Supported`), on the scalar and the 2/4-wide vectorized elementwise cells. Buffers are `unsigned short` (`ushort2`/`ushort4` for a vector cell); the body is computed at `float` (Unpopped's `Compute` rule): each element is widened on load and the result narrowed once on store, round to nearest even. The conversion is Unpopped's plain-C codec `unpopped::cfamily::half_helpers` (checked by `unpopped-cpu-c` against an independent codec), unchanged except `static` becomes `static __device__` and `INFINITY`/`NAN` become `__builtin_inff()`/`__builtin_nanf("")`. **No `__half` or `__hip_bfloat16` and no AMD header beyond `hip_runtime.h`.** Refused, typed: a reduction, scan or any other schedule at a half; a runtime parameter at a half (its dtype is not settled). Tests: `tests/narrow.rs`. **Text only: no HIP compiler has seen it (rig row);** `ldexpf`/`frexpf`/`floorf` on the device are assumed from `hip_runtime.h`. One-off host check, not in CI: the emitted `a * b + a` kernel for f16 and bf16, scalar and vectorized (odd length, so the scalar tail runs), through a shim `hip_runtime.h`, compiled with host clang++ at `-O0` and `-O2` with `-ffp-contract=off`, over every `a` (65,536) against 4,673 operand patterns per `a` (all of 0..2047 and 63489..65535 plus every 97th between, plus one padding element, so subnormals, infinities and NaNs are included), against an independent integer-bit reference codec: 306,249,728 cases per run, 0 mismatches in all eight runs (NaNs compared as a class). A seeded mutant (ties round up in the f16 encoder) was caught: 35,762 mismatches.
## 0.0.34

- `rockfish-hip-emit` and `rockfish-hip-parse` move to Unpopped 0.45.12 and take its version, 0.45.12 (`unpopped` / `unpopped-vocab` `0.45.12`). No source change was needed; every test unchanged and green. The four patent-flagged statements (`AsyncWait`/`CopyAsync`/`AsyncCommit`/`Collective`) are not lowered: the emitter handles `KernelPlan`s only (see `docs/PATENT-SWEEP.md`). **Text only: no HIP compiler has seen it (rig row).**
## 0.0.33

- Docs only: `docs/PATENT-SWEEP.md` records the NVIDIA async-scheduling and barrier-aggregation (`Collective`) patent notices (CireSnave 2026-11-13) and Rockfish's exposure: none today (the emitter lowers `KernelPlan`s, never `Stmt`).
## 0.0.32

- `rockfish-hip-emit` and `rockfish-hip-parse` move to Unpopped 0.43.2 (the newest on crates.io; Unpopped's own main is ahead at 0.45.x, unpublished) and take its version, 0.43.2. `ReduceOp` is now `#[non_exhaustive]` and gained `And`/`Or`/`Xor`/`MaxNum`/`MinNum`/arg-reductions; the emitter spells only `Sum`/`Mean`/`Prod`/`Max`/`Min`, so `check_fold_op` now returns a typed `shape` refusal ("not spelled by this emitter") for any other op instead of failing to compile. Test: `unspelled_reduce_ops_are_refused_not_panicked` (2 of the 5 probed ops reach the emitter; the plan gate refuses the others). The refusal was not mutation-checked. All other tests unchanged and green. **Text only: no HIP compiler has seen it (rig row).**

## 0.0.31

- `rockfish-hip-emit` and `rockfish-hip-parse` move to Unpopped 0.40.11 and take its version, 0.40.11 (`unpopped` / `unpopped-vocab` `0.40.11`). Under `Backend::total_integer_ops` the plan gate now admits an integer `Select` and the six comparisons, and `Hip` spells them at `i32`/`i64`/`u32`/`u64`: a comparison is `((a OP b) ? (T)1 : (T)0)`, `Select` is `((c) != 0 ? (T)(a) : (T)(b))`, one arm evaluated. Integer `Rem` has no native op; it is `Select(b == 0, 0, a - (a / b) * b)` over the total `Div` (`b == 0` gives 0, `MIN % -1` gives 0), and now lowers. Still refused: integer `Max`/`Min`/`Fma`/unary ops (gate or typed decline). Tests: `tests/int_ops.rs` (comparison text for all six ops at four dtypes, Rem text per dtype). **Text only: no HIP compiler has seen it (rig row).** One-off host check, not in CI: the emitted `Rem` kernel text for all four dtypes, through a shim `hip_runtime.h`, was compiled with host clang++ at `-O0` and `-O2` and run over the full cross product of boundary values (0, 1, 2, 3, 7, MAX, MAX-1, MIN, MIN+1, MAX/2, MAX/2+1, signed -1, -2, -3, -7, MIN/2, and 2^53-1/2^53/2^53+1 at 64 bits) against an independent reference (`b == 0` gives 0, `MIN % -1` gives 0, else `%`): 0 mismatches in all eight runs (256/361/121/196 cases per dtype). A seeded mutant (zero-divisor arm gives `a`) was caught (15 mismatches at i32).

## 0.0.30

- `rockfish-hip-emit` and `rockfish-hip-parse` move to Unpopped 0.40.7 and take its version, 0.40.7 (the crates that track Unpopped; dependency `unpopped` / `unpopped-vocab` `0.40.7`). 0.40.7 adds `Backend::total_integer_ops`; `Hip` opts in and now lowers integer compute at `i32`, `i64`, `u32`, `u64` on the scalar elementwise cell. By the total rule of Unpopped's `docs/int-div-shift-semantics.md`: `Div` is a small `__device__` helper (`b == 0 ? a : ...`, signed `MIN / -1` gives `MIN`, the division only in the arm that runs when the guards pass); `Shl`/`Shr` mask the count (`& 31u` / `& 63ull`) and `Shl` goes through the unsigned type; `Add`/`Sub`/`Mul` wrap through the unsigned type when signed; `BitAnd`/`BitOr`/`BitXor` are the C operators; `ConstInt` is an exact literal of the kernel's type (`7LL`, `7u`, `7ull`, the signed minimum as `(-MAX - 1)`), declined when out of range or of another type. Only the helpers a kernel uses are emitted. Refused, typed: an integer vectorized, reduction or scan schedule; every float-only node at an integer dtype. Not spelled: integer `Rem` (the plan gate does not admit the `Select` it needs at an integer dtype). `Hip::supports_dtype` now answers `Supported` for the four integer dtypes. Tests: `tests/int_ops.rs` (text, each guard, refusals). **Text only: no HIP compiler has seen it (rig row).** One-off host check, not in CI: the emitted helper bodies for all four dtypes were compiled with host clang++ at `-O0` and `-O2` and matched an independent reference on the boundary vectors (divisors `0, 1, -1, MIN, MAX`-class, counts `0, 1, N-1, N, N+1, 255, -1`); a mutant without the zero guard crashed on the host's trapping divide. A mutant with the wrong mask width was NOT caught on the host (x86 masks shift counts in hardware), so the mask is pinned by the text test only.

## 0.0.29

- `rockfish-hip-emit` (stays 0.40.0, tracking Unpopped): the `Hip` backend now lowers `f64` as well as `f32` on a `rocm:` target, across all four shapes (scalar, vectorized, reduction, scan). At `f64` operands and outputs are `double` (`double2` / `double4` vectors), math calls drop the `f` suffix (`sqrt`, `fma`, `fmax`), a constant is printed as the unsuffixed `f64` (non-finite still refused), `Select` tests against `0.0`, and the kernel is named `rockfish_hip_<op>_f64`. The `f32` text is unchanged. Not done, deliberately: total integer `Div`/`Shl`/`Shr`. Unpopped 0.40 rejects an integer `Div` at the plan gate and admits integer dtypes only with `const_int`/`int_op` seams, so those spellings are a separate integer-dtype row, not a change to this float path. **Text only: no HIP compiler has seen it (rig row).**
## 0.0.28

- `rockfish-hip-emit` (stays 0.40.0, tracking Unpopped): the `Hip` backend spells three more `f32` nodes it used to refuse. `Max`/`Min` are the NaN-propagating compare-select with A on ties (`a != a ? a : (b != b ? b : (a >= b ? a : b))`), never `fmaxf`/`fminf` (those are `FmaxIeee`/`FminIeee`). `Fma` is `fmaf(a, b, c)`, one rounding. `Select` tests the condition against `0.0f` in f32 and picks an arm without computing on it. `rockfish-hip-parse` lifts neither `fmaf` nor a C cast, so Fma and Select are pinned by a text test; Max/Min lift to their compare-select expansion (test with a `>=` to `>` control). **Text only: no HIP compiler has seen it (rig row).**

## 0.0.27

- `rockfish-hip-emit` and `rockfish-hip-parse` move to Unpopped 0.40.0 (Breaking, pre-1.0: the second number) and take its version, 0.40.0, as the crates that track Unpopped. The dependency is `unpopped` / `unpopped-vocab` `0.40`. **No source change was needed:** 0.40.0's one addition, `Stmt::AtomicRmw` (relaxed, global integer buffer element), is in the kernel statement layer, and Rockfish lowers `KernelPlan`s only; nothing here matches on `Stmt`, so there is no arm to add and no atomic is emitted (a kernel-layer HIP emitter would spell `atomicAdd` / `atomicMax` / `atomicMin` on `int`; that row does not exist yet). Unpopped 0.39 oracle changes (a window with no in-bounds tap panics; `Shl`/`Shr` counts are taken modulo the width) do not touch the text emitter. Re-verified: workspace clippy `-D warnings` clean and every test green with `--all-features`.

## 0.0.26

- `rockfish-hip-emit` (stays 0.38.4, tracking Unpopped): the `Hip` backend now lowers a body that reads runtime scalar parameters (`Param(i)`), where it used to refuse it. Every kernel (scalar, vectorized, reduction, scan) takes the params it reads as trailing `float p<i>` arguments after its last fixed argument, ascending by index, distinct, unread indices omitted; the body spells the leaf `p<i>`. In a vectorized kernel a param stays a scalar `float` used by every lane. The argument order is a stated launch contract. Params are `f32` only, as Unpopped's plan gate already requires. Tests: 3 new in `emit.rs`, 1 in `fold.rs` (the old "a param is refused" test is replaced). **Text only: no HIP compiler has seen it (rig row).** Also: the vectorized emitter's string literals are written with `\n` escapes instead of raw line breaks (same text).

## 0.0.25

- Docs only: `ROADMAP.md` status cells match what merged. R5 reads merged with R5-H as its own open item (needs a second AMD machine, board 329); R6c (`rockfish-miopen-sys`, #15) and R6-L (#17) no longer say "in PR"; R7 lists R7c, R7d and the `Vectorized` schedule as merged. No code change.

## 0.0.24

- `rockfish-hip-emit` (stays 0.38.4, tracking Unpopped): the `Hip` backend lowers `Schedule::Vectorized` for the `f32` elementwise cell on a `rocm:` target, widths 2 and 4 (`float2` / `float4` loads and stores, each lane the scalar body over that lane's components, a scalar tail for `n % width`). Other widths are a typed `UnsupportedPlanShape`. Operand alignment is a stated contract of the cell key, not checked. 4 new tests. **Text only: no HIP compiler has seen it (rig row).**

## 0.0.23

- Test only: `rockfish-core/tests/no_text_file_contains_a_nul_byte.rs`, a port of Unpopped's guard. It scans `git ls-files` for a 0 byte in any text-extension file, with a population control and a matcher control, and was seen red on an injected NUL. Clean at `origin/main` (0 offenders). No library change.

## 0.0.22

- `rockfish-hip-emit` (new feature `hiprtc`, off by default; runtime-loaded, no ROCm needed to build): `HipRtcCompiler`, an `unpopped::Compiler` over `rockfish-hiprtc` that compiles HIP source to an AMD GPU code object with `--offload-arch=<gfx>` and reports `ArtifactKind::HipCodeObject` (Unpopped 0.38.0+). `HipRtcCompiler::new` declines, as a typed `HipRtcTargetError`, a target outside `rocm:` or whose capability set is not `gfx<lowercase hex>`; a compile failure returns hipRTC's log in the error string; `max_compile_ms` is not enforced (hipRTC has no deadline API). Tests: 4 (target grammar with both decline arms, artifact kind, and a compile that passes both ways, printing which). **hipRTC compile UNVERIFIED: this laptop has no `hiprtc.dll` (measured: the compile test took the absent-library path), so the option spelling and the produced bytes have never run; rig row.**


## 0.0.21

- `rockfish-hip-emit`: the `Hip` backend now also lowers the sequential last-axis reduction (`Sum`, `Mean`, `Prod`, `Max`, `Min`) and the serial prefix scan (`Sum`, `Prod`, `Max`, `Min`; `reverse`, `exclusive`), `f32` on a `rocm:` target, as one-thread-per-row kernels with a `double` accumulator (the oracle's width) and the launch ABI `(in0.., out, rows, k)`. Outer/middle/multi-axis reductions, fused post-expressions, the cooperative block scan, `RowReduce`, `Window`, `Contraction` and `RowSort` stay typed `LowerError`s. No wavefront width is used or assumed (the emitter owns the width; `KernelPlan` carries none). Tests: 11 new (goldens, per-op text with a pairwise-different control, flag effects, each refusal paired with a lowering control, a no-warp-assumption check). One-off host check (VS `clang++` with a HIP built-in shim, not in CI): 21 kernels match a reference transcription of the oracle loops. **Still no HIP compiler has seen any emitted text, and nothing ran on a GPU (compilation UNVERIFIED; rig row).**

## 0.0.20

- New crate `rockfish-hip-emit` (unpublished; version tracks Unpopped, 0.38.4 = the `unpopped` built and tested against): the `Hip` `unpopped::backend::Backend` (`name` "hip", `provider` "rockfish"), so Unpopped can lower a plan to HIP C++ source. Scope is one cell: the scalar contiguous elementwise schedule, one output, `f32`, on a `rocm:` target; everything else (vectorized/strided schedules, runtime parameters, other dtypes, other namespaces, `Fma`, `Select`, comparisons, `Max`/`Min` and the ops listed in `docs/DESIGN-R7.md` section 5a) is a typed `LowerError`. Spellings are stated in the emitter, not inherited from CUDA output, and are the inverse of what `rockfish-hip-parse` lifts; constants are rounded to `f32` as the IR defines and refused if they overflow, underflow or are non-finite; the kernel body turns floating-point contraction off. Tests: golden text, round trip through the HIP frontend with a one-operator-change control, and each refusal paired with a lowering control. **Emits text only: no HIP compiler has seen it and nothing ran on a GPU (compilation UNVERIFIED; rig row).** `KernelPlan` carries no wavefront size (recorded for R7c).
- `Cargo.lock`: `unpopped` and `unpopped-vocab` 0.38.0 -> 0.38.4; `rockfish-hip-parse`'s 14 tests pass against 0.38.4 (its own version stays 0.38.0 until the PM allocates).

## 0.0.19

- New crate `rockfish-hip-parse` (unpublished; version tracks Unpopped, 0.38.0 = the `unpopped` built against): `parse_hip`, `HIP_RESIDUE` and the `HIP` `unpopped::convert::Frontend`, so HIP C++ elementwise, reduction and scan kernels lift to Unpopped's IR. Grammar is `tree-sitter-cuda` 0.21 (no HIP grammar is on crates.io); 11 hand-written HIP idioms (HIP index macros, `__launch_bounds__`, `__restrict__`, `extern "C"`, `__device__` helpers, `<<<>>>`, `hipLaunchKernelGGL`, `__half`) parse with no error node. Residue (shared memory, atomics, barriers, wave/lane intrinsics, AMDGCN builtins, BLAS/MIOpen calls, host launches, asm) refuses as `Inexpressible`; fast-math spellings (`__expf`) refuse rather than lift as the accurate op. Text-level only: no HIP code was compiled or run, and only these idioms are tested.

## 0.0.18

- Docs only: new `docs/DESIGN-R7.md` (HIP C++ parser and emitter on Unpopped's `Frontend`/`Backend`/`Compiler`; `rockfish-hip-parse` and `rockfish-hip-emit` track Unpopped's version; nothing compiled or run on a GPU) and a ROADMAP pointer. No code change.

## 0.0.17

- Docs only: `docs/DESIGN-R5.md` records the decision that `hip()` is a process-wide lazily loaded table (as Baracuda's `driver()`), and corrects the section 3 and 5 text that described the superseded `Arc<Hip>` handle and fake-table test. No code change.

## 0.0.16

- `rockfish-core` (Windows): a library candidate given as an absolute path (`ROCKFISH_<NAME>_PATH`, or a found install) is now loaded with `LOAD_WITH_ALTERED_SEARCH_PATH`, with forward slashes rewritten to backslashes, so DLLs it depends on are found beside it. Measured on one Windows laptop, same DLL, directory not on `PATH`: Ollama's ROCm 7.1 `rocblas.dll` failed to load before (`LoadLibraryExW failed`) and loads after (56 symbols resolve, version `5.1.1.787bb67272-dirty`), also when the path is written `C:\Users\...\Local/Programs/...`. Bare names keep the default search. Unchanged on Linux. The sibling-dependency case is measured by hand only; the tests cover the forward-slash path and a missing path.

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
