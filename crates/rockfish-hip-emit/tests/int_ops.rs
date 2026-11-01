//! Integer `Div`, `Shl`, `Shr` and `ConstInt` in `rockfish-hip-emit` (the total rule of
//! Unpopped's `docs/int-div-shift-semantics.md`).
//!
//! **What these show:** the text the emitter produces for each of `i32`/`i64`/`u32`/`u64`,
//! that every guard of the rule is present in it, and that the opt-in is declared and what stays
//! refused. **What they do not show:** that a HIP compiler accepts it or that a GPU computes the
//! rule's results: no HIP compiler ran (rig row). The helper bodies were separately compiled by
//! a host C++ compiler and run on the boundary vectors; that check is not repeated here.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{BinaryOp, Expr, OpDef, ScalarExpr, input};
use unpopped::plan::{Schedule, try_build_plan_for};
use unpopped::try_generate;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key};

const INTS: [(ElementKind, &str, &str, &str); 4] = [
    (ElementKind::I32, "i32", "int", "31u"),
    (ElementKind::I64, "i64", "long long", "63ull"),
    (ElementKind::U32, "u32", "unsigned int", "31u"),
    (ElementKind::U64, "u64", "unsigned long long", "63ull"),
];

fn key(n_in: usize, dt: ElementKind, align: u32) -> StructureKey {
    let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
    let cat = if n_in == 1 {
        OpCategory::UnaryElementwise
    } else {
        OpCategory::BinaryElementwise
    };
    structure_key(
        cat,
        &vec![a; n_in + 1],
        TargetId::parse("rocm:gfx906").expect("target"),
    )
}

fn src(o: &OpDef, dt: ElementKind) -> String {
    try_generate(o, &key(o.n_inputs as usize, dt, 4), &Hip)
        .unwrap_or_else(|e| panic!("{} at {dt:?}: {e:?}", o.name))
        .source
}

fn lower_err(o: &OpDef, dt: ElementKind) -> LowerError {
    let k = key(o.n_inputs as usize, dt, 4);
    let plan = try_build_plan_for(o, &k, &Hip).expect("plan");
    Hip.lower(&plan).expect_err("must refuse")
}

fn bin(name: &str, dt: ElementKind, f: impl Fn(Expr, Expr) -> Expr) -> OpDef {
    OpDef::elementwise(name, 2, &[dt], f(input(0), input(1)))
}

#[test]
fn the_backend_opts_in_to_the_total_integer_rule() {
    assert!(Hip.total_integer_ops());
}

#[test]
fn div_is_the_total_rule_at_every_integer_dtype() {
    for (dt, tag, c, _) in INTS {
        let s = src(&bin("idiv", dt, |a, b| a / b), dt);
        assert!(
            s.contains(&format!("rockfish_div_{tag}(in0[i], in1[i])")),
            "{tag}: call\n{s}"
        );
        assert!(
            s.contains(&format!(
                "static __device__ __forceinline__ {c} rockfish_div_{tag}({c} a, {c} b)"
            )),
            "{tag}: definition\n{s}"
        );
        // The zero guard, with the division in the arm that runs only when it passes.
        assert!(s.contains("b == 0 ? a : "), "{tag}: zero guard\n{s}");
        let min_guard = match tag {
            "i32" => Some("a == (-2147483647 - 1) && b == -1"),
            "i64" => Some("a == (-9223372036854775807LL - 1) && b == -1"),
            _ => None,
        };
        match min_guard {
            Some(g) => assert!(s.contains(g), "{tag}: MIN / -1 guard\n{s}"),
            None => assert!(
                !s.contains("== -1"),
                "{tag}: unsigned has no MIN / -1 guard"
            ),
        }
        assert!(s.contains("a / b"), "{tag}: the division itself\n{s}");
        assert!(
            s.contains(&format!("rockfish_hip_idiv_{tag}")),
            "{tag}: kernel name"
        );
        assert!(
            s.contains(&format!("const {c}* in0,")),
            "{tag}: operand type"
        );
    }
}

#[test]
fn shifts_mask_the_count_and_shift_left_goes_through_unsigned() {
    for (dt, tag, c, mask) in INTS {
        let shl = src(&bin("sl", dt, |a, b| a.binary(BinaryOp::Shl, b)), dt);
        let shr = src(&bin("sr", dt, |a, b| a.binary(BinaryOp::Shr, b)), dt);
        assert!(shl.contains(&format!("rockfish_shl_{tag}(in0[i], in1[i])")));
        assert!(shr.contains(&format!("rockfish_shr_{tag}(in0[i], in1[i])")));
        let (u, expect_shl, expect_shr) = match tag {
            "i32" | "i64" => {
                let u = if tag == "i32" {
                    "unsigned int"
                } else {
                    "unsigned long long"
                };
                (
                    u,
                    format!("({c})(({u})a << (({u})b & {mask}))"),
                    format!("a >> (({u})b & {mask})"),
                )
            }
            _ => (
                c,
                format!("a << (b & {mask})"),
                format!("a >> (b & {mask})"),
            ),
        };
        let _ = u;
        assert!(shl.contains(&expect_shl), "{tag}: shl\n{shl}");
        assert!(shr.contains(&expect_shr), "{tag}: shr\n{shr}");
        // Only the helper a kernel uses is emitted.
        assert!(!shl.contains(&format!("rockfish_shr_{tag}")), "{tag}");
        assert!(!shl.contains(&format!("rockfish_div_{tag}")), "{tag}");
    }
}

#[test]
fn add_sub_mul_wrap_through_the_unsigned_type_when_signed() {
    let s = src(&bin("a", ElementKind::I32, |a, b| a + b), ElementKind::I32);
    assert!(
        s.contains("((int)((unsigned int)in0[i] + (unsigned int)in1[i]))"),
        "{s}"
    );
    let s = src(&bin("m", ElementKind::I64, |a, b| a * b), ElementKind::I64);
    assert!(
        s.contains("((long long)((unsigned long long)in0[i] * (unsigned long long)in1[i]))"),
        "{s}"
    );
    let s = src(&bin("s", ElementKind::U32, |a, b| a - b), ElementKind::U32);
    assert!(s.contains("(in0[i] - in1[i])"), "{s}");
    // No helper when the body uses none.
    assert!(!s.contains("rockfish_div"), "{s}");
}

fn konst_int(v: i64, dt: ElementKind) -> Expr {
    Expr(ScalarExpr::ConstInt(v, dt))
}

#[test]
fn const_int_is_an_exact_literal_of_the_kernel_type() {
    let cases = [
        (ElementKind::I32, 7, "7"),
        (ElementKind::I32, -5, "(-5)"),
        (ElementKind::I32, i64::from(i32::MIN), "(-2147483647 - 1)"),
        (ElementKind::I32, i64::from(i32::MAX), "2147483647"),
        (ElementKind::I64, 7, "7LL"),
        (ElementKind::I64, -5, "(-5LL)"),
        (ElementKind::I64, i64::MIN, "(-9223372036854775807LL - 1)"),
        (ElementKind::I64, i64::MAX, "9223372036854775807LL"),
        (ElementKind::U32, 7, "7u"),
        (ElementKind::U32, i64::from(u32::MAX), "4294967295u"),
        (ElementKind::U64, 7, "7ull"),
        (ElementKind::U64, i64::MAX, "9223372036854775807ull"),
    ];
    for (dt, v, lit) in cases {
        let o = OpDef::elementwise("c", 1, &[dt], input(0) + konst_int(v, dt));
        let s = src(&o, dt);
        assert!(s.contains(lit), "{dt:?} {v}: want `{lit}`\n{s}");
    }
}

#[test]
fn a_const_int_that_does_not_fit_or_has_the_wrong_type_is_declined() {
    let bad = [
        (ElementKind::I32, i64::from(i32::MAX) + 1),
        (ElementKind::I32, i64::from(i32::MIN) - 1),
        (ElementKind::U32, -1),
        (ElementKind::U32, i64::from(u32::MAX) + 1),
        (ElementKind::U64, -1),
    ];
    for (dt, v) in bad {
        let o = OpDef::elementwise("c", 1, &[dt], input(0) + konst_int(v, dt));
        let e = lower_err(&o, dt);
        assert!(
            matches!(e, LowerError::UnsupportedOp { .. }),
            "{dt:?} {v}: {e:?}"
        );
    }
}

#[test]
fn what_the_integer_cell_does_not_spell_stays_refused() {
    // An integer `Max`/`Min` is still not admitted at an integer compute dtype by the plan gate.
    let o = bin("mx", ElementKind::I32, |a, b| a.binary(BinaryOp::Max, b));
    assert!(try_build_plan_for(&o, &key(2, ElementKind::I32, 4), &Hip).is_err());
    // A vectorized integer cell is a typed schedule refusal, not float vectors over ints.
    let o = bin("v", ElementKind::I32, |a, b| a / b);
    let k = key(2, ElementKind::I32, 256);
    let plan = try_build_plan_for(&o, &k, &Hip).expect("plan");
    if matches!(plan.schedule, Schedule::Vectorized { .. }) {
        assert!(matches!(
            Hip.lower(&plan),
            Err(LowerError::UnsupportedSchedule { .. })
        ));
    } else {
        panic!("expected a vectorized cell, got {:?}", plan.schedule);
    }
    // 8- and 16-bit integer division stay refused at the plan gate.
    for dt in [ElementKind::I8, ElementKind::U8, ElementKind::I16] {
        let o = bin("d8", dt, |a, b| a / b);
        assert!(
            try_build_plan_for(&o, &key(2, dt, 4), &Hip).is_err(),
            "{dt:?}"
        );
    }
}

#[test]
fn comparisons_are_one_or_zero_in_the_compute_type() {
    for (dt, tag, c, _) in INTS {
        for (op, c_op) in [
            (BinaryOp::CmpEq, "=="),
            (BinaryOp::CmpNe, "!="),
            (BinaryOp::CmpLt, "<"),
            (BinaryOp::CmpLe, "<="),
            (BinaryOp::CmpGt, ">"),
            (BinaryOp::CmpGe, ">="),
        ] {
            let s = src(&bin("cmp", dt, |a, b| a.binary(op, b)), dt);
            let want = format!("((in0[i] {c_op} in1[i]) ? ({c})1 : ({c})0)");
            assert!(s.contains(&want), "{tag} {c_op}: want `{want}`\n{s}");
        }
    }
}

/// The remainder of `docs/int-div-shift-semantics.md`: `Select(b == 0, 0, a - (a / b) * b)`
/// over the total `Div`. `b == 0` gives 0 (not `a`), and `MIN % -1` is 0 because `MIN / -1`
/// wraps to `MIN`.
fn rem(dt: ElementKind) -> OpDef {
    let (a, b) = (input(0), input(1));
    let zero = konst_int(0, dt);
    let body = b
        .clone()
        .binary(BinaryOp::CmpEq, zero.clone())
        .select(zero, a.clone() - (a / b.clone()) * b);
    OpDef::elementwise("irem", 2, &[dt], body)
}

fn zero_lit(tag: &str) -> &'static str {
    match tag {
        "i32" => "0",
        "i64" => "0LL",
        "u32" => "0u",
        _ => "0ull",
    }
}

#[test]
fn rem_is_a_select_over_the_total_div_at_every_integer_dtype() {
    for (dt, tag, c, _) in INTS {
        let s = src(&rem(dt), dt);
        assert!(
            s.contains(&format!(
                "((in1[i] == {}) ? ({c})1 : ({c})0)",
                zero_lit(tag)
            )),
            "{tag}: condition\n{s}"
        );
        assert!(s.contains("!= 0 ? "), "{tag}: select tests != 0\n{s}");
        let call = format!("rockfish_div_{tag}(in0[i], in1[i])");
        assert!(s.contains(&call), "{tag}: total Div inside the Rem\n{s}");
        // The call sits in the kernel body, after the select's condition (the helper's own
        // definition above does not call itself).
        let sel = s.find("!= 0 ? ").expect("select");
        let at = s.rfind(&call).expect("call");
        assert!(
            sel < at,
            "{tag}: the division is in an arm of the select\n{s}"
        );
    }
}
