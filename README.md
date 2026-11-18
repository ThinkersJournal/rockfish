# rockfish

Rust interface to AMD ROCm/HIP. The crate names `rockfish` and `rockfish-sys` are reserved on crates.io at 0.0.0 (no release yet). The workspace has a HIP runtime loader and safe layer, hipRTC, rocBLAS/hipBLASLt/MIOpen bindings, and a HIP C++ parser and emitter on Unpopped's contract. What was measured, on one Windows laptop, and what is UNVERIFIED (kernel launch, hipRTC compile, Linux, GPU GEMM) is in `ROADMAP.md`.

## Declined by name

The emitter never lowers NVIDIA-specific statements to hardware operations on AMD hardware: `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` on an mbarrier (NVIDIA US11803380B2), and `WgmmaWait`, a wait on a count of pending matrix-multiply operations (NVIDIA US12204897B2 and its continuation US12710961B2); CireSnave's rulings of 2026-11-16. `Tcgen05Commit` is on the same list (US11803380B2 and CireSnave's mbarrier ruling, extended to it by the PM; not read separately against the claims). Also declined by name (board 434 C; CireSnave, 2026-11-13): `AsyncWait`, `CopyAsync`, `AsyncCommit` (US12118382B2, US11294713B2) and `Collective`, `CollectiveTuple` (US8539204B2). Builders of non-NVIDIA emitters: lowering these to a hardware operation risks infringing those patents; do not. Every kernel-layer kernel is refused with a typed `UnsupportedKernel` error, and a kernel containing one of these statements is refused by name; see `docs/PATENT-SWEEP.md`.

Licensed under MIT or Apache-2.0, at your option.
