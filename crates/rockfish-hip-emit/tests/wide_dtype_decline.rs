// SPDX-License-Identifier: MIT OR Apache-2.0
//! The seven `ElementKind` rows added in Unpopped 0.60.0 (KISS 0.24.0) are declined by name.
//!
//! **What this shows:** (a) `Hip::supports_dtype`, Hip's own answer, is `Unsupported` for each of
//! `F128`, `I128`, `U128`, `I256`, `U256`, `F8E3M4` and `F8UE4M3` on a `rocm:` target; (b) generating
//! an elementwise kernel at that dtype is refused by Unpopped's planner before Hip is reached
//! (proves the planner, not Hip); the control dtype `f32` on the same path succeeds. **What it does not show:** what HIP offers for these types.
//! UNVERIFIED and unread: no HIP header was consulted. This emitter spells only the eight dtypes
//! in `Hip::supports_dtype`; a 128-bit integer is not spelled by any lowering here (clang's
//! `__int128` is a host-side extension, not claimed for device code), and no 8-bit float type
//! is spelled either. Nothing here says HIP cannot; only that this emitter does not.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, DtypeSupport};
use unpopped::ir::{OpDef, input};
use unpopped::try_generate;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

const WIDE: [ElementKind; 7] = [
    ElementKind::F128,
    ElementKind::I128,
    ElementKind::U128,
    ElementKind::I256,
    ElementKind::U256,
    ElementKind::F8E3M4,
    ElementKind::F8UE4M3,
];

fn generate(dt: ElementKind) -> Result<String, String> {
    let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 1);
    let k = structure_key(
        OpCategory::BinaryElementwise,
        &[a, a, a],
        TargetId::parse("rocm:gfx906").unwrap(),
    );
    let o = OpDef::elementwise("add", 2, &[dt], input(0) + input(1));
    try_generate(&o, &k, &Hip)
        .map(|g| g.source)
        .map_err(|e| format!("{e:?}"))
}

#[test]
fn hip_own_answer_is_unsupported_for_each_wide_dtype() {
    // (a) Hip's OWN answer, from its allow-list. Mutant (run 2026-11-26): adding `F128` to the
    // `Supported` arm of `Hip::supports_dtype` makes this test fail.
    let rocm = TargetId::parse("rocm:gfx906").unwrap();
    for dt in WIDE {
        assert_eq!(
            Hip.supports_dtype(dt, rocm),
            DtypeSupport::Unsupported,
            "{dt:?}"
        );
    }
}

#[test]
fn generation_is_refused_planner_refusal_unpopped_not_a_hip_decline() {
    // (b) PLANNER refusal (Unpopped, `InadmissibleOpInCell`), before Hip is reached: it proves
    // the planner, not Hip.
    for dt in WIDE {
        generate(dt).expect_err(&format!("{dt:?} must be refused"));
    }
}

#[test]
fn control_f32_on_the_same_path_generates() {
    let s = generate(ElementKind::F32).expect("f32 control");
    assert!(s.contains("rockfish_hip_add_f32"), "{s}");
}
