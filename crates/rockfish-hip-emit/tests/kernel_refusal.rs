// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refusal of kernel-layer kernels in `rockfish-hip-emit`, and by name of the declined list.
//!
//! **What this shows:** `Hip` lowers no kernel-layer (`KernelFn`) kernel; it answers the typed
//! `LowerError::UnsupportedKernel`, never a panic and never text. A kernel containing one of
//! NVIDIA's statements declined on AMD hardware (`MbarInit`, `MbarArrive`, `MbarCompleteTx`,
//! `MbarWait`, `CopyBulk` on an mbarrier: US11803380B2; `WgmmaWait`: US12204897B2 and
//! US12710961B2; `Tcgen05Commit`: US11803380B2 by the PM's extension; CireSnave's rulings of
//! 2026-11-16), and the async-scheduling and barrier-aggregation statements `AsyncWait`,
//! `CopyAsync`, `AsyncCommit`, `Collective` and `CollectiveTuple` (US12118382B2, US11294713B2,
//! US8539204B2; board 434 C, CireSnave 2026-11-13), is refused **by name**, at any nesting depth. See `docs/PATENT-SWEEP.md`.
//! **What it does not show:** that no other path lowers them; there is none: `lower` takes plans.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, CacheHint, CollectiveKind, KernelFn, OperandLayout, Place, Stmt, Unroll,
};
use unpopped_vocab::ElementKind;

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "refusal_control",
        ElementKind::F32,
        Vec::new(),
        Vec::new(),
        Block(body),
    )
}

fn place() -> Place {
    Place::Alloc {
        alloc: 0,
        index: ScalarExpr::Const(0.0),
    }
}

fn refusal(body: Vec<Stmt>) -> String {
    match Hip.lower_kernel(&kernel(body)) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

fn declined() -> Vec<(&'static str, Stmt, &'static str)> {
    vec![
        (
            "MbarInit",
            Stmt::MbarInit {
                bar: place(),
                count: ScalarExpr::Const(1.0),
            },
            "US11803380B2",
        ),
        (
            "MbarArrive",
            Stmt::MbarArrive {
                bar: place(),
                tx: None,
            },
            "US11803380B2",
        ),
        (
            "MbarCompleteTx",
            Stmt::MbarCompleteTx {
                bar: place(),
                bytes: ScalarExpr::Const(1.0),
            },
            "US11803380B2",
        ),
        (
            "MbarWait",
            Stmt::MbarWait {
                bar: place(),
                parity: ScalarExpr::Const(0.0),
            },
            "US11803380B2",
        ),
        (
            "CopyBulk",
            Stmt::CopyBulk {
                dst: place(),
                src: place(),
                elems: 1,
                bar: place(),
            },
            "US11803380B2",
        ),
        ("WgmmaWait", Stmt::WgmmaWait { pending: 0 }, "US12204897B2"),
        ("AsyncWait", Stmt::AsyncWait { pending: 0 }, "US12118382B2"),
        (
            "CopyAsync",
            Stmt::CopyAsync {
                dst: place(),
                src: place(),
                elems: 1,
                valid: None,
                cache: CacheHint::default(),
                l2_prefetch: None,
            },
            "US11294713B2",
        ),
        ("AsyncCommit", Stmt::AsyncCommit, "US12118382B2"),
        (
            "Collective",
            Stmt::Collective {
                kind: CollectiveKind::Reduce(ReduceOp::Sum),
                value: ScalarExpr::Const(1.0),
                result: 0,
                scope: Scope::Subgroup,
            },
            "US8539204B2",
        ),
        (
            "CollectiveTuple",
            Stmt::CollectiveTuple {
                kind: CollectiveKind::Reduce(ReduceOp::Sum),
                values: vec![ScalarExpr::Const(1.0)],
                results: vec![0],
                scope: Scope::Subgroup,
            },
            "US8539204B2",
        ),
        (
            "Tcgen05Commit",
            Stmt::Tcgen05Commit { bar: place() },
            "US11803380B2",
        ),
        (
            "CopyUnpack",
            Stmt::CopyUnpack {
                dst: place(),
                src: place(),
                groups: 1,
                bar: place(),
            },
            "US11803380B2",
        ),
    ]
}

#[test]
fn a_kernel_layer_kernel_is_refused_with_the_typed_error() {
    let detail = refusal(Vec::new());
    assert!(detail.contains("refusal_control"), "{detail}");
    assert!(
        !detail.contains("declined"),
        "control: no declined name: {detail}"
    );
}

#[test]
fn every_declined_statement_is_refused_by_name_with_its_patent() {
    for (name, stmt, patent) in declined() {
        let detail = refusal(vec![stmt]);
        assert!(detail.contains(&format!("`{name}`")), "{name}: {detail}");
        assert!(detail.contains(patent), "{name}: {detail}");
        assert!(detail.contains("declined"), "{name}: {detail}");
    }
}

#[test]
fn a_declined_statement_is_found_inside_if_for_and_while() {
    for (name, stmt, _) in declined() {
        let nested = [
            Stmt::If {
                cond: ScalarExpr::Const(1.0),
                then: Block(Vec::new()),
                els: Block(vec![stmt.clone()]),
            },
            Stmt::For {
                local: 0,
                start: ScalarExpr::Const(0.0),
                end: ScalarExpr::Const(1.0),
                step: ScalarExpr::Const(1.0),
                unroll: Unroll::None,
                body: Block(vec![stmt.clone()]),
            },
            Stmt::While {
                cond: ScalarExpr::Const(1.0),
                body: Block(vec![stmt]),
            },
        ];
        for outer in nested {
            let detail = refusal(vec![outer]);
            assert!(detail.contains(&format!("`{name}`")), "{name}: {detail}");
        }
    }
}

/// Unpopped 0.57.0 additions this emitter has no lowering for, refused by name with a reason
/// (no patent ruling: the text says "no lowering", never "declined").
fn unlowered() -> Vec<(&'static str, Stmt)> {
    let raw = Place::Raw {
        buffer: 0,
        byte_offset: ScalarExpr::Const(0.0),
        elem: ElementKind::F32,
    };
    let araw = Place::AllocRaw {
        alloc: 0,
        byte_offset: ScalarExpr::Const(0.0),
        elem: ElementKind::F32,
    };
    let canonical = OperandLayout::Canonical {
        major: unpopped::kernel::Major::MN,
        swizzle: unpopped::kernel::Swizzle::B128,
        lbo: Some(16),
        sbo: 32,
    };
    vec![
        (
            "ProxyFence",
            Stmt::ProxyFence {
                space: MemorySpace::Shared,
            },
        ),
        (
            "Wgmma",
            Stmt::Wgmma {
                n: 8,
                ab: ElementKind::F16,
                acc: ElementKind::F32,
                a: place(),
                b: place(),
                d: 0,
                scale_d: ScalarExpr::Const(0.0),
                a_layout: canonical,
                b_layout: canonical,
                a_regs: None,
            },
        ),
        (
            "Wgmma",
            Stmt::Wgmma {
                n: 8,
                ab: ElementKind::F16,
                acc: ElementKind::F32,
                a: place(),
                b: place(),
                d: 0,
                scale_d: ScalarExpr::Const(0.0),
                a_layout: OperandLayout::default(),
                b_layout: canonical,
                a_regs: Some(1),
            },
        ),
        ("WgmmaCommit", Stmt::WgmmaCommit),
        (
            "Tcgen05Mma",
            Stmt::Tcgen05Mma {
                n: 8,
                ab: ElementKind::F16,
                acc: ElementKind::F32,
                a: place(),
                b: place(),
                d: 0,
                d_col: ScalarExpr::Const(0.0),
                scale_d: ScalarExpr::Const(0.0),
                a_layout: canonical,
                b_layout: canonical,
                m: 64,
                d_lane: 16,
                b_ty: Some(ElementKind::Fp8E5M2),
                negate_a: true,
                negate_b: false,
            },
        ),
        ("Tcgen05Alloc", Stmt::Tcgen05Alloc { d: 0, ncols: 32 }),
        ("Tcgen05Dealloc", Stmt::Tcgen05Dealloc { d: 0, ncols: 32 }),
        ("Tcgen05RelinquishPermit", Stmt::Tcgen05RelinquishPermit),
        (
            "Tcgen05Ld",
            Stmt::Tcgen05Ld {
                src: 0,
                col: ScalarExpr::Const(0.0),
                count: 1,
                dst: 1,
            },
        ),
        (
            "Place::Raw",
            Stmt::Load {
                local: 0,
                place: raw.clone(),
            },
        ),
        (
            "Place::Raw",
            Stmt::Store {
                place: raw,
                value: ScalarExpr::Const(0.0),
            },
        ),
        (
            "Place::AllocRaw",
            Stmt::Load {
                local: 0,
                place: araw.clone(),
            },
        ),
        (
            "Place::AllocRaw",
            Stmt::Store {
                place: araw,
                value: ScalarExpr::Const(0.0),
            },
        ),
    ]
}

#[test]
fn the_0_57_additions_are_refused_by_name_with_a_reason_at_any_depth() {
    for (name, stmt) in unlowered() {
        let direct = refusal(vec![stmt.clone()]);
        assert!(direct.contains(&format!("`{name}`")), "{name}: {direct}");
        assert!(direct.contains("no lowering"), "{name}: {direct}");
        assert!(
            !direct.contains("declined"),
            "{name}: no patent ruling is claimed: {direct}"
        );
        let nested = refusal(vec![Stmt::If {
            cond: ScalarExpr::Const(1.0),
            then: Block(vec![stmt]),
            els: Block(Vec::new()),
        }]);
        assert!(
            nested.contains(&format!("`{name}`")),
            "nested {name}: {nested}"
        );
    }
}

/// A kernel the Unpopped checker accepts: one declared `Tensor` allocation of 32 columns, a
/// one-dimensional launch of 32-lane subgroups, and the three 0.58.0 tensor-memory allocation
/// statements. The refusal tests above use placeholder operands; this one shows the by-name
/// refusal also fires on a kernel `check_kernel` passes, so it is not a checker error in disguise.
fn tmem_alloc_kernel(body: Vec<Stmt>) -> KernelFn {
    let mut k =
        kernel(body).with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    k.allocs = vec![AllocDecl::new(
        "tmem",
        MemorySpace::Tensor,
        ElementKind::F32,
        Extent::Static(128 * 32),
    )];
    k
}

#[test]
fn the_0_58_tensor_memory_allocation_statements_are_refused_on_a_checker_valid_kernel() {
    let k = tmem_alloc_kernel(vec![
        Stmt::Tcgen05Alloc { d: 0, ncols: 32 },
        Stmt::Tcgen05Dealloc { d: 0, ncols: 32 },
        Stmt::Tcgen05RelinquishPermit,
    ]);
    unpopped::kernel::check_kernel(&k).expect("the fixture is checker-valid");
    for (name, stmt) in [
        ("Tcgen05Alloc", Stmt::Tcgen05Alloc { d: 0, ncols: 32 }),
        ("Tcgen05Dealloc", Stmt::Tcgen05Dealloc { d: 0, ncols: 32 }),
        ("Tcgen05RelinquishPermit", Stmt::Tcgen05RelinquishPermit),
    ] {
        let one = tmem_alloc_kernel(vec![stmt]);
        match Hip.lower_kernel(&one) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(detail.contains(&format!("`{name}`")), "{name}: {detail}");
                assert!(detail.contains("no lowering"), "{name}: {detail}");
            }
            other => panic!("{name}: expected UnsupportedKernel, got {other:?}"),
        }
    }
}
