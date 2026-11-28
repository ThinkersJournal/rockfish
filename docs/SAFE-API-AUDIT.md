# Safe-API size/slice audit (rockfish-hip, rockfish-rocblas)

Read at `origin/main` 206a9e1 (2026-11-28); tests added on this branch. Method: read every safe
`pub fn` that takes a slice, buffer or size, name the check between the argument and the FFI call,
then, per function family, a short-by-one call that must return `Err(Misuse)` before any FFI, with
the unshortened call as control, and a red-first mutant (the check removed; the test must fail as a
value difference). **A "check present" read is not a measurement** (the aocl lane's lesson); the
status column says which rows are which. No host guard-page harness applies: these crates pass
**device** buffers whose length `DeviceBuffer` carries. The rocBLAS tests use a NULL handle
(`Handle::null_for_test`) and no rocBLAS library: a call that got past a guard fails with
`LibraryNotFound`, not `Misuse`, so a removed guard changes the error kind and fails the test.

All tests below are GPU-labelled (one HIP device for the buffers; measured on the Radeon 610M,
gfx1036, HIP 6.4; without a device they print so and stop; never `#[ignore]`).

| Safe fn | Guard before the FFI call | Test | Status |
|---|---|---|---|
| `copy_from_host` | `same_len` | `copies_short_by_one_are_refused_and_move_nothing` | MEASURED, mutant caught (1) |
| `copy_to_host` | `same_len` | same test (sentinel survives) | MEASURED, mutant caught (1) |
| `copy_to_pinned_async` | `same_len` + device equality | same test (sentinel survives) | MEASURED, mutant caught (1) |
| `copy_from_pinned_async` | `same_len` + device equality | `async_copies_round_trip_through_pinned_memory` | MEASURED, mutant caught (1) |
| `copy_from_device` | `same_len` + same-device | `device_to_device_copy_and_fill_zero` | MEASURED, mutant caught (1) |
| `DeviceBuffer::alloc` / `PinnedBuffer::alloc` | zero and `len * size_of` overflow refused | `zero_and_overflowing_sizes_are_refused_before_hip_is_asked` | test exists; NO mutant run |
| `Handle::gemm` | `plan()`, `need()` per buffer A, B, C | `gemm_refuses_a_buffer_short_by_one_before_rocblas` (+ `plan_*`) | MEASURED, 3 mutants caught |
| `Handle::gemm_strided_batched` | `plan()`, `batched_len` per buffer, `stride_c >= c_len` | `gemm_strided_batched_refuses_a_buffer_short_by_one_before_rocblas` (+ `batched_len_*`) | MEASURED, 4 mutants caught |
| `fill_zero`, `to_vec` | no caller-supplied length | n/a | READ-ONLY |
| `Function::launch` | `unsafe`: contract in its doc; HIP cannot check arguments against the signature | not checkable | READ-ONLY |
| `rockfish-hiprtc`: `Program::new`, `add_name_expression`, `lowered_name`, `compile` | no caller-supplied length: every string goes through `c_string` (NUL refused), the option count through `c_int::try_from`; `log()` and `code()` size their buffers from hipRTC's own size query | `c_string` unit tests only | READ 2026-11-28, UNMEASURED: no `hiprtc.dll` on this box, so no call has reached hipRTC; the buffer sizes are library-trusted |
| rocBLAS level 1/2 | no such wrapper exists (`gemm` and `gemm_strided_batched` only) | n/a | n/a |

(1) Mutants: each guard call replaced by nothing (copy_from_host, copy_to_host, copy_to_pinned_async,
copy_from_pinned_async, copy_from_device: 5 mutants, each anchored once, each failed its test on a
value difference). rocBLAS mutants: `need(a|b|c, p.x_len,` to `need(.., 0,`; `batched_len(p.x_len, ..)`
to `batched_len(0, ..)` for A, B, C; `batch > 1 && stride_c < p.c_len` to `false`: 7 mutants, each
anchored once, each failed with "expected Misuse, got Err(Load(LibraryNotFound ..))". Directions
were chosen so a missing guard copies fewer bytes than the larger side holds (never out of bounds).

## Not measured

- **No GEMM has run on any device** (`UNVERIFIED on a GPU`); these tests prove the guards refuse,
  not that an accepted GEMM is right. rocBLAS never ran in these tests (null handle, no library).
- The `alloc` guard has no mutant run. `Function::launch` and `fill_zero`/`to_vec` are READ-ONLY.
- rockfish-hiprtc is in the table above as READ-ONLY (no library on this box); rocFFT, rocRAND, rocSOLVER,
  rocSPARSE, MIOpen (`-sys` crates only, no library here). UNMEASURED.

— lane: rockfish
