//! Refusal control for kernel-layer kernels in `rockfish-hip-emit`.
//!
//! **What this shows:** `Hip` does not lower kernel-layer (`KernelFn`) kernels; it answers the
//! typed `LowerError::UnsupportedKernel`, never a panic and never text. That default is what
//! declines NVIDIA's mbarrier statements (`MbarInit`, `MbarArrive`, `MbarCompleteTx`,
//! `MbarWait`, `CopyBulk` on an mbarrier; US11803380B2) and `WgmmaWait` (US12204897B2), CireSnave's rulings of 2026-11-16 on AMD
//! hardware today: see `docs/PATENT-SWEEP.md`. **What it does not show:** a refusal *by name*
//! of those statements. `unpopped` 0.47.0 (the pin) has no `Mbar*` variant to build one with;
//! when a release that has them is pinned, add one test per statement beside this one, and a
//! by-name arm in `Hip::lower_kernel`.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::kernel::{Block, KernelFn};
use unpopped_vocab::ElementKind;

#[test]
fn a_kernel_layer_kernel_is_refused_with_the_typed_error() {
    let kernel = KernelFn::new(
        "refusal_control",
        ElementKind::F32,
        Vec::new(),
        Vec::new(),
        Block::default(),
    );
    match Hip.lower_kernel(&kernel) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(
                detail.contains("refusal_control"),
                "the refusal names the kernel: {detail}"
            );
        }
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}
