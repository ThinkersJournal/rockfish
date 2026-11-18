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
use unpopped::exec::Scope;
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{Block, CacheHint, CollectiveKind, KernelFn, Place, Stmt, Unroll};
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
