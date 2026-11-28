# `Function::launch` read audit (rockfish-hip)

Read at `origin/main` 4eaa5e5, 2026-11-28, `crates/rockfish-hip/src/module.rs` (`launch`, `LaunchConfig`,
`KernelArgs`), `stream.rs`, `buffer.rs`, `repr.rs`. **A read, not a harness run:** no kernel has been
launched by this crate on any device (no `hiprtc.dll`, no code object, no GPU test calls `launch`; positive
control: `grep -rn "launch" crates/rockfish-hip/src` finds only `module.rs`). Every "holds" below is
UNMEASURED.

| # | What `launch` does | Result of the read |
|---|---|---|
| 1 | `config.validate()` | Refuses a zero in any of the six dimensions (unit test `zero_dimensions_are_refused`). Nothing else is checked here: block size, grid size and `shared_bytes` against device limits are left to the runtime, whose answer is an `Error::Api` (UNMEASURED: no device was asked). |
| 2 | `stream.device() != module.device` | Refused as `Misuse`. No test reaches it (needs two devices). |
| 3 | Argument list | `KernelArgs::value` stores a pointer to the caller's value, `buffer` the address of the buffer's pointer field (`ptr_slot`); both borrow for `'a`, so the values outlive the `launch` call. That HIP copies the argument bytes at enqueue (so the borrow need not outlast the call) is UNVERIFIED here: it is the premise of the borrow lifetime and of the doc line "borrowed until the launch call returns". |
| 4 | Argument count, order, size | Not checkable: `hipModuleLaunchKernel` takes untyped pointers, which is why `launch` is `unsafe`. A wrong count or size is undefined behaviour in the runtime. The doc contract states this. |
| 5 | Lifetime of buffers and stream after the call | The call is asynchronous; `launch` borrows `&Stream` and `&KernelArgs` only for the call. Nothing in the type system ties a `DeviceBuffer` or the `Stream` to the in-flight work. The doc contract puts that on the caller. `Stream` and `Event` have `Drop` impls that destroy the handle; whether `hipStreamDestroy` waits for queued work is UNVERIFIED (runtime behaviour, not read). |
| 6 | `T: DeviceRepr` bound on `value` | `DeviceRepr` is an `unsafe` trait (`Copy + 'static`, no padding, no pointers). A struct passed by value must match the kernel's parameter layout: stated contract, not checked. |

**Decision recorded:** `launch` stays `unsafe` with its doc contract; no safe wrapper is added until a
typed signature exists. The lowered kernels' launch ABI (pointer plus per-axis `long long` strides) is the
same kind of stated contract (HANDOFF fact), not checked.

**Open (needs a device or a library, not a decision):** items 1, 3 and 5 above. First device row: a GPU test
that launches a trivial code object, labelled local, never `#[ignore]`d (ruling board 329).

— lane: rockfish
