//! Sequential integer reductions in `rockfish-hip-emit` (`Sum`/`Prod`/`Max`/`Min` at
//! `i32`/`i64`/`u32`/`u64`).
//!
//! **What these show:** the text, that `Sum`/`Prod` accumulate in the unsigned type of the same
//! width (so signed overflow is never undefined), that `Max`/`Min` compare in the dtype with an
//! empty axis answering 0, and what stays refused. **What they do not show:** that a HIP
//! compiler accepts it (rig row). The emitted kernels were run on the host as a one-off (see the
//! CHANGELOG); that check is not repeated here.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::plan::{Schedule, try_build_plan_for};
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key};

const INTS: [(ElementKind, &str, &str, &str); 4] = [
    (ElementKind::I32, "i32", "int", "unsigned int"),
    (ElementKind::I64, "i64", "long long", "unsigned long long"),
    (ElementKind::U32, "u32", "unsigned int", "unsigned int"),
    (
        ElementKind::U64,
        "u64",
        "unsigned long long",
        "unsigned long long",
    ),
];

fn key(dt: ElementKind) -> StructureKey {
    let mat = OperandDesc::new(2, &[64, 1024], &[1024, 1], dt, 4);
    let row = OperandDesc::new(1, &[64], &[1], dt, 4);
    structure_key(
        OpCategory::Reduction,
        &[mat, row],
        TargetId::parse("rocm:gfx906").expect("target"),
    )
}

fn lower(dt: ElementKind, op: ReduceOp) -> Result<String, LowerError> {
    let o = OpDef::reduction("red", 1, &[dt], input(0), op);
    let k = key(dt);
    let plan = try_build_plan_for(&o, &k, &Hip).expect("plan");
    assert!(
        matches!(plan.schedule, Schedule::Reduction { .. }),
        "{:?}",
        plan.schedule
    );
    Hip.lower(&plan).map(|k| k.source)
}

#[test]
fn sum_and_prod_accumulate_in_the_unsigned_type() {
    for (dt, tag, t, u) in INTS {
        let sum = lower(dt, ReduceOp::Sum).unwrap();
        assert!(
            sum.contains(&format!("void rockfish_hip_red_{tag}(")),
            "{sum}"
        );
        assert!(sum.contains(&format!("{u} acc = ({u})0;")), "{sum}");
        assert!(
            sum.contains(&format!("{t} v = ({t})(in0[r * k + j]);")),
            "{sum}"
        );
        assert!(sum.contains(&format!("acc = acc + ({u})v;")), "{sum}");
        assert!(sum.contains(&format!("out[r] = ({t})acc;")), "{sum}");
        assert!(!sum.contains("double"), "{sum}");
        let prod = lower(dt, ReduceOp::Prod).unwrap();
        assert!(prod.contains(&format!("{u} acc = ({u})1;")), "{prod}");
        assert!(prod.contains(&format!("acc = acc * ({u})v;")), "{prod}");
    }
}

#[test]
fn max_and_min_compare_in_the_dtype_and_an_empty_axis_answers_zero() {
    for (dt, _, t, _) in INTS {
        let mx = lower(dt, ReduceOp::Max).unwrap();
        assert!(mx.contains(&format!("{t} acc = 0;")), "{mx}");
        assert!(mx.contains("bool have = false;"), "{mx}");
        assert!(mx.contains("if (!have || v > acc) { acc = v; }"), "{mx}");
        assert!(mx.contains("have = true;"), "{mx}");
        let mn = lower(dt, ReduceOp::Min).unwrap();
        assert!(mn.contains("if (!have || v < acc) { acc = v; }"), "{mn}");
        // No NaN rule at an integer dtype.
        assert!(!mx.contains("v != v") && !mn.contains("v != v"));
    }
}

#[test]
fn what_an_integer_reduction_does_not_spell_is_a_typed_refusal() {
    for (dt, ..) in INTS {
        for op in [ReduceOp::Mean, ReduceOp::And, ReduceOp::Or, ReduceOp::Xor] {
            let o = OpDef::reduction("red", 1, &[dt], input(0), op);
            // The plan gate may refuse first; when it builds, the emitter must refuse.
            let k = key(dt);
            if let Ok(plan) = try_build_plan_for(&o, &k, &Hip) {
                let e = Hip.lower(&plan).expect_err("must be refused");
                assert!(
                    matches!(e, LowerError::UnsupportedPlanShape { .. }),
                    "{op:?}: {e:?}"
                );
            }
        }
    }
}
