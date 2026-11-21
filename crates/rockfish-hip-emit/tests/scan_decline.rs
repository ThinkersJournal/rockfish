//! Unpopped's typed decline of a scan over a non-dense operand (unpopped 0.56.5): the plan gate
//! answers `PlanError::InadmissibleOpInCell` instead of panicking, so a caller's
//! `(OpDef, structure_key)` pair is refused before it reaches this emitter, and there is no strided
//! scan for `rockfish-hip-emit` to spell. Each refusal has a control that plans and lowers.

use rockfish_hip_emit::Hip;
use unpopped::backend::Backend;
use unpopped::ir::{OpDef, ReduceOp, input, reduced};
use unpopped::plan::{PlanError, try_build_plan_for};
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key};

fn key(in_strides: &[i64]) -> StructureKey {
    let dt = ElementKind::F32;
    let input = OperandDesc::new(2, &[64, 1024], in_strides, dt, 4);
    let out = OperandDesc::new(2, &[64, 1024], &[1024, 1], dt, 4);
    structure_key(
        OpCategory::Scan,
        &[input, out],
        TargetId::parse("rocm:gfx906").expect("target"),
    )
}

fn scan() -> OpDef {
    OpDef::scan(
        "scn",
        1,
        &[ElementKind::F32],
        ReduceOp::Sum,
        1,
        false,
        false,
        input(0),
        reduced(0),
    )
}

#[test]
fn a_scan_over_a_transposed_input_is_a_typed_plan_error_not_a_panic() {
    // Control: the dense layout plans and lowers.
    let (op, dense_key) = (scan(), key(&[1024, 1]));
    let dense = try_build_plan_for(&op, &dense_key, &Hip).expect("dense plans");
    assert!(Hip.lower(&dense).is_ok());
    // A transposed input (last axis not unit-stride) is declined by name.
    let bad_key = key(&[1, 64]);
    match try_build_plan_for(&op, &bad_key, &Hip) {
        Err(PlanError::InadmissibleOpInCell { detail }) => {
            assert!(!detail.is_empty(), "the decline carries its reason");
        }
        other => panic!("expected InadmissibleOpInCell, got {other:?}"),
    }
}
