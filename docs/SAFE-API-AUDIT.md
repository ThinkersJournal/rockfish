# Safe-API size/slice audit (rockfish-hip, rockfish-rocblas)

Read at `origin/main` 206a9e1 (2026-11-28). Method: read every safe `pub fn` that takes a slice,
buffer or size, and name the check that stands between the argument and the FFI call, plus the test
that exercises it. The aocl `safe-api-audit` guard-page harness targets host slices handed to a C
library; these crates pass **device** buffers whose length is carried by `DeviceBuffer`, so the
guard is a length comparison, not a page fault, and no host guard-page harness applies.

| Safe fn | Guard before the FFI call | Test |
|---|---|---|
| `DeviceBuffer::copy_from_host` / `copy_to_host` | `same_len(self.len, data.len())` (exact equality) | `length_mismatch_is_refused` (helper), `host_device_round_trip_...` (GPU) |
| `copy_from_pinned_async` / `copy_to_pinned_async` | `same_len` + stream/buffer device equality | `async_copies_round_trip_through_pinned_memory` (GPU) |
| `copy_from_device` | `same_len` + same-device check | `device_to_device_copy_and_fill_zero` (GPU) |
| `fill_zero`, `to_vec` | length is the buffer's own | same GPU test |
| `DeviceBuffer::alloc` / `PinnedBuffer::alloc` | zero and `len * size_of` overflow refused before HIP | `zero_and_overflowing_sizes_are_refused_before_hip_is_asked` |
| `Handle::gemm` | `plan()` (32-bit dims, `ld >= rows`, checked span arithmetic), `need()` per buffer, same-device | `plan_*` (7 tests, host-only) |
| `Handle::gemm_strided_batched` | as `gemm` + `batched_len` checked arithmetic, `stride_c >= c_len` when `batch > 1`, 63-bit strides | `batched_len_counts_strides_between_matrices_only`, `plan_*` |
| `Function::launch` | `unsafe`: contract stated in its doc (HIP cannot check arguments against the signature) | not checkable |

## Result

No safe function reaches an FFI call with an unchecked length. Nothing was changed.

## Not measured

- `gemm`/`gemm_strided_batched` have never run on a device (`UNVERIFIED on a GPU`); the guards are
  proven by the host-only `plan_*` tests, not by a GEMM that would fault on a short buffer.
- Whether the `need()` calls themselves are exercised end to end (a short `DeviceBuffer` passed to
  `gemm`) is not tested; only `plan()` and `batched_len()` are. Row for a GPU-labelled test.
- rockfish-hiprtc (not read in this pass; no `hiprtc.dll` on this box), rocFFT, rocRAND, rocSOLVER, rocSPARSE, MIOpen (`-sys` crates only, no library here). UNMEASURED.

— lane: rockfish
