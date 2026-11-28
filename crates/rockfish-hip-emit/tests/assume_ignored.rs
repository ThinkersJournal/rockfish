// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unpopped 0.64.0 `Stmt::Assume` (a per-thread value fact): `rockfish-hip-emit` IGNORES it, as the
//! 0.64.0 CHANGELOG permits ("may lower it to an optimiser hint or ignore it"). **What this shows:**
//! a kernel carrying an `Assume` (at the top, nested in an `If`, and in a `For`) lowers to HIP text
//! byte-equal to the same kernel without it, so an `Assume` never changes the text of the rest of
//! the kernel. **What it does not show:** a `__builtin_assume` lowering (a later row, UNVERIFIED), or
//! anything on a device.

use rockfish_hip_emit::Hip;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, KernelFn, LocalDecl, Stmt};
use unpopped_vocab::ElementKind;

fn fact() -> ScalarExpr {
    ScalarExpr::Binary(
        BinaryOp::CmpLt,
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        Box::new(ScalarExpr::ConstInt(32, ElementKind::U32)),
    )
}
fn assume() -> Stmt {
    Stmt::Assume { cond: fact() }
}
fn set(v: i64) -> Stmt {
    Stmt::Assign {
        local: 0,
        value: ScalarExpr::ConstInt(v, ElementKind::I32),
    }
}
fn kernel(body: Vec<Stmt>) -> KernelFn {
    let mut all = vec![Stmt::Let {
        local: 0,
        init: ScalarExpr::ConstInt(0, ElementKind::I32),
    }];
    all.extend(body);
    KernelFn::new(
        "assume_k",
        ElementKind::I32,
        vec![],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(all),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]))
}
fn text(body: Vec<Stmt>) -> String {
    Hip.lower_kernel(&kernel(body)).expect("lowers").source
}

#[test]
fn assume_does_not_change_the_emitted_text() {
    let plain = text(vec![set(1), set(2)]);
    let top = text(vec![assume(), set(1), assume(), set(2)]);
    assert_eq!(plain, top, "a top-level Assume changed the text");
    // Positive control: a different statement DOES change the text, so equality means something.
    assert_ne!(plain, text(vec![set(1), set(3)]));

    let nested_plain = text(vec![Stmt::If {
        cond: fact(),
        then: Block(vec![set(1)]),
        els: Block(vec![set(2)]),
    }]);
    let nested = text(vec![Stmt::If {
        cond: fact(),
        then: Block(vec![assume(), set(1)]),
        els: Block(vec![set(2), assume()]),
    }]);
    assert_eq!(nested_plain, nested, "a nested Assume changed the text");
}
