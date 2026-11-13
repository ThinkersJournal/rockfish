//! Serial integer prefix scans in `rockfish-hip-emit` (`Sum`/`Prod`/`Max`/`Min` at
//! `i32`/`i64`/`u32`/`u64`).
//!
//! **What these show:** the text, that `Sum`/`Prod` accumulate in the unsigned type of the same
//! width, that `Max`/`Min` compare in the dtype and an exclusive scan writes the dtype's extreme
//! (the oracle's `scan_identity`) until the first element is folded, and what stays refused.
//! **What they do not show:** that a HIP compiler accepts it (rig row).

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{OpDef, ReduceOp, input, reduced};
use unpopped::plan::{Schedule, try_build_plan_for};
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key};

/// (dtype, tag, C type, unsigned C type, Max's exclusive identity, Min's exclusive identity)
const INTS: [(ElementKind, &str, &str, &str, &str, &str); 4] = [
    (
        ElementKind::I32,
        "i32",
        "int",
        "unsigned int",
        "(-2147483647 - 1)",
        "2147483647",
    ),
    (
        ElementKind::I64,
        "i64",
        "long long",
        "unsigned long long",
        "(-9223372036854775807LL - 1)",
        "9223372036854775807LL",
    ),
    (
        ElementKind::U32,
        "u32",
        "unsigned int",
        "unsigned int",
        "0",
        "4294967295u",
    ),
    (
        ElementKind::U64,
        "u64",
        "unsigned long long",
        "unsigned long long",
        "0",
        "18446744073709551615ull",
    ),
];

fn key(dt: ElementKind) -> StructureKey {
    let mat = OperandDesc::new(2, &[64, 1024], &[1024, 1], dt, 4);
    structure_key(
        OpCategory::Scan,
        &[mat, mat],
        TargetId::parse("rocm:gfx906").expect("target"),
    )
}

fn lower(
    dt: ElementKind,
    op: ReduceOp,
    reverse: bool,
    exclusive: bool,
) -> Result<String, LowerError> {
    let o = OpDef::scan(
        "scn",
        1,
        &[dt],
        op,
        1,
        reverse,
        exclusive,
        input(0),
        reduced(0),
    );
    let k = key(dt);
    let plan = try_build_plan_for(&o, &k, &Hip).expect("plan");
    assert!(
        matches!(plan.schedule, Schedule::Scan { .. }),
        "{:?}",
        plan.schedule
    );
    Hip.lower(&plan).map(|k| k.source)
}

#[test]
fn sum_and_prod_scan_in_the_unsigned_type() {
    for (dt, tag, t, u, ..) in INTS {
        let sum = lower(dt, ReduceOp::Sum, false, false).unwrap();
        assert!(
            sum.contains(&format!("void rockfish_hip_scn_{tag}(")),
            "{sum}"
        );
        assert!(sum.contains(&format!("{u} acc = ({u})0;")), "{sum}");
        assert!(sum.contains(&format!("acc = acc + ({u})v;")), "{sum}");
        assert!(
            sum.contains(&format!("out[r * k + j] = ({t})acc;")),
            "{sum}"
        );
        assert!(!sum.contains("double"), "{sum}");
        let prod = lower(dt, ReduceOp::Prod, false, false).unwrap();
        assert!(prod.contains(&format!("{u} acc = ({u})1;")), "{prod}");
        assert!(prod.contains(&format!("acc = acc * ({u})v;")), "{prod}");
    }
}

#[test]
fn inclusive_stores_after_the_fold_and_exclusive_before_it() {
    for (dt, _, t, ..) in INTS {
        let store = format!("out[r * k + j] = ({t})acc;");
        let inc = lower(dt, ReduceOp::Sum, false, false).unwrap();
        assert!(
            inc.find("acc = acc + ").unwrap() < inc.find(&store).unwrap(),
            "{inc}"
        );
        let exc = lower(dt, ReduceOp::Sum, false, true).unwrap();
        assert!(
            exc.find(&store).unwrap() < exc.find("acc = acc + ").unwrap(),
            "{exc}"
        );
        let rev = lower(dt, ReduceOp::Sum, true, false).unwrap();
        assert!(rev.contains("long long j = k - 1 - t;"), "{rev}");
        assert!(inc.contains("long long j = t;"), "{inc}");
    }
}

#[test]
fn max_and_min_compare_in_the_dtype() {
    for (dt, _, t, ..) in INTS {
        let mx = lower(dt, ReduceOp::Max, false, false).unwrap();
        assert!(mx.contains(&format!("{t} acc = 0;")), "{mx}");
        assert!(mx.contains("bool have = false;"), "{mx}");
        assert!(mx.contains("if (!have || v > acc) { acc = v; }"), "{mx}");
        assert!(!mx.contains("double") && !mx.contains("inf"), "{mx}");
        let mn = lower(dt, ReduceOp::Min, false, false).unwrap();
        assert!(mn.contains("if (!have || v < acc) { acc = v; }"), "{mn}");
    }
}

#[test]
fn exclusive_max_and_min_write_the_dtype_extreme_until_the_first_fold() {
    for (dt, _, t, _, max_id, min_id) in INTS {
        let mx = lower(dt, ReduceOp::Max, false, true).unwrap();
        assert!(
            mx.contains(&format!("out[r * k + j] = have ? acc : ({t})({max_id});")),
            "{mx}"
        );
        let mn = lower(dt, ReduceOp::Min, true, true).unwrap();
        assert!(
            mn.contains(&format!("out[r * k + j] = have ? acc : ({t})({min_id});")),
            "{mn}"
        );
    }
}

#[test]
fn mean_and_bitwise_scans_emit_nothing() {
    // Unpopped's plan gate asserts (panics) on an integer `Mean` scan and the oracle rejects the
    // bitwise ops, so the refusal is upstream of the emitter; if a future gate lets one through,
    // the emitter must still refuse it. Either way, no source comes out.
    for (dt, ..) in INTS {
        for op in [ReduceOp::Mean, ReduceOp::And, ReduceOp::Or, ReduceOp::Xor] {
            let emitted = std::panic::catch_unwind(|| {
                let o = OpDef::scan("scn", 1, &[dt], op, 1, false, false, input(0), reduced(0));
                let k = key(dt);
                let plan = try_build_plan_for(&o, &k, &Hip).ok()?;
                Hip.lower(&plan).ok()
            });
            assert!(!matches!(emitted, Ok(Some(_))), "{dt:?} {op:?} emitted");
        }
    }
}
