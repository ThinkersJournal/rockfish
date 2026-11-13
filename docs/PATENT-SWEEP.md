# Patent sweep (self-serve; not legal advice; no attorney)

## Async scheduling statements (CireSnave ruling 2026-11-13, board 431 hit 22)

Unpopped publishes `Stmt::AsyncWait` / `CopyAsync` / `AsyncCommit` with a risk note. CireSnave: *"it applies to any code created for non-Nvidia hardware that executes a hardware instruction that causes async scheduling of instructions in the hardware itself. Without knowing what Nividia's competitors' hardware have for instructions, I don't know whether any of them contain instructions that infringe on Nvidia's claims."*

Patents US12118382B2 / US11294713B2 (NVIDIA). Claim 1, verbatim: *"prevent one or more first instructions of a first thread from being performed while one or more second instructions of the first thread are being performed"*. Claims 9/17/23 are the same as method/system/processor. Dependents: the wait is on output of other threads in shared memory.

**Rockfish exposure at origin/main (2026-11-13): none.** `rockfish-hip-emit` lowers `KernelPlan`s only (scalar, `Vectorized`, sequential reduction, serial scan); it never matches on `Stmt`, so it cannot receive `AsyncWait`/`CopyAsync`/`AsyncCommit`. A search of `crates/*/src` and `crates/*/tests` for `Stmt`, `AsyncWait`, `CopyAsync`, `AsyncCommit`, `cp.async`, `global_load_lds`, `s_waitcnt`, `__builtin_amdgcn` finds 0 hits; the same search for `KernelPlan` finds 15 in `rockfish-hip-emit/src/lib.rs`, so the query works. Emitted text contains no async-copy or wait instruction. The hipRTC path compiles that text and adds none.

**Before any kernel-layer (`Stmt`) emitter row for AMD:** list the target's async instructions (candidates to check, UNVERIFIED: LDS-direct/async global loads and `s_waitcnt`-style counters on gfx9/gfx11/gfx12), record them here, and prefer a typed decline by name over lowering these statements. Nobody has checked AMD's instruction set against the claims.

## Barrier aggregation, `Stmt::Collective` (CireSnave ruling 2026-11-13, board 431 hit 23 A: publish with risk note)

US8539204B2 (NVIDIA), claim 1: receive a barrier aggregation instruction for a first thread; combine its value with an aggregation result; communicate the result to each thread; wait until all threads have received it before executing more instructions. Claim 13 adds a hardware "barrier instruction execution unit". Unpopped's oracle only simulates it; the exposure is an emitter lowering Collective Reduce/Scan to a hardware barrier-with-reduction instruction.

**Rockfish exposure at origin/main (2026-11-13): none.** The emitter never matches on `Stmt`, so it cannot receive `Collective`. 0 hits in `crates/*/src` and `crates/*/tests` for `Collective`, `barrier`, `__syncthreads`, `__builtin_amdgcn_s_barrier`, `__builtin_amdgcn_ds_*`, `wave_barrier`; control: `KernelPlan` has 15 hits in `rockfish-hip-emit/src/lib.rs`. Reductions and scans are emitted as a sequential single-thread fold per row (no inter-thread combine, no barrier). **Before a block-parallel reduce/scan row** (HANDOFF's next-rows list): that row would introduce `__syncthreads()`/LDS combines or wave intrinsics; list them here first, and do not lower a `Collective` to a barrier-with-reduction instruction without a ruling (typed decline by name is the default). Nobody has checked AMD's instruction set against the claim.
