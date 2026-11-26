// SPDX-License-Identifier: MIT OR Apache-2.0
//! Audit, by name, of the six `IntOp` variants Unpopped 0.56.x added (`Div`, `Shl`, `Shr`,
//! `BitAnd`, `BitOr`, `BitXor`; Unpopped notice #386).
//!
//! **What this shows:** `rockfish-hip-emit` never receives an `IntOp`: it leaves the `int_op` seam
//! unset and spells integers through the `BinaryOp`/`ArithOp` seams. This test names each of the six
//! variants (`IntOp` is `non_exhaustive`, so the matches below keep a wildcard that panics: a
//! variant added later is NOT caught by the compiler, only by an audit like this one) and pins the source each one reaches at every integer dtype.
//! **What it does not show:** that a HIP compiler accepts the text (rig row).

use rockfish_hip_emit::Hip;
use unpopped::backend::IntOp;
use unpopped::ir::{BinaryOp, Expr, OpDef, input};
use unpopped::try_generate;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key};

const INTS: [(ElementKind, &str); 4] = [
    (ElementKind::I32, "i32"),
    (ElementKind::I64, "i64"),
    (ElementKind::U32, "u32"),
    (ElementKind::U64, "u64"),
];

fn key(dt: ElementKind) -> StructureKey {
    let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 4);
    structure_key(
        OpCategory::BinaryElementwise,
        &vec![a; 3],
        TargetId::parse("rocm:gfx906").expect("target"),
    )
}

fn src(e: impl Fn(Expr, Expr) -> Expr, dt: ElementKind) -> String {
    let o = OpDef::elementwise("audit", 2, &[dt], e(input(0), input(1)));
    try_generate(&o, &key(dt), &Hip)
        .unwrap_or_else(|er| panic!("{dt:?}: {er:?}"))
        .source
}

/// The spelling expected for `op` at the dtype tagged `tag`, by name.
fn expected(op: IntOp, tag: &str) -> String {
    match op {
        IntOp::Div => format!("rockfish_div_{tag}(in0[i], in1[i])"),
        IntOp::Shl => format!("rockfish_shl_{tag}(in0[i], in1[i])"),
        IntOp::Shr => format!("rockfish_shr_{tag}(in0[i], in1[i])"),
        IntOp::BitAnd => "(in0[i] & in1[i])".to_string(),
        IntOp::BitOr => "(in0[i] | in1[i])".to_string(),
        IntOp::BitXor => "(in0[i] ^ in1[i])".to_string(),
        // The other variants are not part of this audit (audited in int_ops.rs).
        _ => unreachable!("not one of the six"),
    }
}

fn build(op: IntOp) -> impl Fn(Expr, Expr) -> Expr {
    move |a, b| match op {
        IntOp::Div => a / b,
        IntOp::Shl => a.binary(BinaryOp::Shl, b),
        IntOp::Shr => a.binary(BinaryOp::Shr, b),
        IntOp::BitAnd => a.binary(BinaryOp::BitAnd, b),
        IntOp::BitOr => a.binary(BinaryOp::BitOr, b),
        IntOp::BitXor => a.binary(BinaryOp::BitXor, b),
        _ => unreachable!("not one of the six"),
    }
}

#[test]
fn each_of_the_six_new_int_ops_is_spelled_by_name_at_every_integer_dtype() {
    let six = [
        IntOp::Div,
        IntOp::Shl,
        IntOp::Shr,
        IntOp::BitAnd,
        IntOp::BitOr,
        IntOp::BitXor,
    ];
    for op in six {
        for (dt, tag) in INTS {
            let s = src(build(op), dt);
            let want = expected(op, tag);
            assert!(s.contains(&want), "{op:?} at {tag}: missing `{want}`\n{s}");
        }
    }
}
