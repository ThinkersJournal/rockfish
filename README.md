# rockfish

Rust interface to AMD ROCm/HIP. The crate names `rockfish` and `rockfish-sys` are reserved on crates.io at 0.0.0 (no release yet). The workspace has a HIP runtime loader and safe layer, hipRTC, rocBLAS/hipBLASLt/MIOpen bindings, and a HIP C++ parser and emitter on Unpopped's contract. What was measured, on one Windows laptop, and what is UNVERIFIED (kernel launch, hipRTC compile, Linux, GPU GEMM) is in `ROADMAP.md`.

## Declined by name

The emitter never lowers NVIDIA-specific mbarrier statements to hardware operations on AMD hardware: `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` on an mbarrier, and `WgmmaWait` while it is on hold (NVIDIA US11803380B2; CireSnave's ruling of 2026-11-16). Today every kernel-layer kernel is refused with a typed `UnsupportedKernel` error; see `docs/PATENT-SWEEP.md`.

Licensed under MIT or Apache-2.0, at your option.
