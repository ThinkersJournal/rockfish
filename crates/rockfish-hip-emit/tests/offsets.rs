// SPDX-License-Identifier: MIT OR Apache-2.0
//! Runtime base offsets on the strided elementwise cell of [`rockfish_hip_emit::Hip`].
//!
//! **What these show.** The emitted text: the `off{i}`/`offo` launch arguments and where they sit
//! (after `so_*`, before `n`), the pointer bumps at kernel entry (before a fully broadcast
//! operand's hoisted load), the kernel-name suffix, and that an op without offsets is unchanged.
//! **What they do not show:** that the text compiles under a HIP compiler or addresses the right
//! elements on a GPU (rig rows, `docs/DESIGN-R7.md` section 6). The argument order and names are
//! those `baracuda-cuda-emit` freezes.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{BaseOffset, Expr, OpDef, input};
use unpopped::plan::{Schedule, try_build_plan};
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

fn gfx906() -> TargetId {
    TargetId::parse("rocm:gfx906").unwrap()
}

fn desc(dt: ElementKind, shape: &[i64], strides: &[i64]) -> OperandDesc {
    let esz = match dt {
        ElementKind::F64 => 8,
        ElementKind::F16 | ElementKind::Bf16 => 2,
        _ => 4,
    };
    OperandDesc::new(shape.len(), shape, strides, dt, esz * 64)
}

const DENSE: (&[i64], &[i64]) = (&[4, 3], &[3, 1]);
const ALL_BCAST: (&[i64], &[i64]) = (&[4, 3], &[0, 0]);

/// Lower a two-input elementwise op into a dense `[4, 3]` output with the given offsets.
fn lower(
    dt: ElementKind,
    body: Expr,
    ins: [(&[i64], &[i64]); 2],
    offs: Option<(Vec<BaseOffset>, BaseOffset)>,
) -> Result<String, LowerError> {
    let mut o = OpDef::elementwise("off", 2, &[dt], body);
    if let Some((i, out)) = offs {
        o = o.with_base_offsets(i, out);
    }
    let mut d: Vec<OperandDesc> = ins.iter().map(|(s, st)| desc(dt, s, st)).collect();
    d.push(desc(dt, &[4, 3], &[3, 1]));
    let key = structure_key(OpCategory::BinaryElementwise, &d, gfx906());
    let plan = try_build_plan(&o, &key).expect("plan");
    assert_eq!(plan.schedule, Schedule::Strided, "an offset forces Strided");
    Hip.lower(&plan).map(|k| k.source)
}

fn run(offs: Option<(Vec<BaseOffset>, BaseOffset)>) -> String {
    lower(ElementKind::F32, input(0) + input(1), [DENSE, DENSE], offs).unwrap()
}

#[test]
fn input_and_output_offsets_add_arguments_bumps_and_a_name_suffix() {
    let got = run(Some((
        vec![BaseOffset::Zero, BaseOffset::Runtime],
        BaseOffset::Runtime,
    )));
    assert!(got.contains(" void rockfish_hip_off_f32_off1o(\n"), "{got}");
    assert!(
        got.contains(
            "    long long so_1,\n    long long off1,\n    long long offo,\n    long long n)"
        ),
        "off1, offo sit after the output strides and before n:\n{got}"
    );
    assert!(
        got.contains("    #pragma clang fp contract(off)\n    in1 += off1;\n    out += offo;\n"),
        "{got}"
    );
    assert!(!got.contains("off0"), "input 0 has no offset:\n{got}");
}

#[test]
fn a_bump_precedes_the_hoisted_load_of_a_fully_broadcast_operand() {
    let got = lower(
        ElementKind::F32,
        input(0) + input(1),
        [DENSE, ALL_BCAST],
        Some((
            vec![BaseOffset::Zero, BaseOffset::Runtime],
            BaseOffset::Zero,
        )),
    )
    .unwrap();
    let bump = got.find("    in1 += off1;\n").expect("bump");
    let hoist = got.find("const float h1 = in1[0];").expect("hoist");
    assert!(bump < hoist, "{got}");
    assert!(got.contains("rockfish_hip_off_f32_off1("), "{got}");
    assert!(!got.contains("offo"), "{got}");
}

#[test]
fn without_offsets_the_kernel_is_unchanged() {
    // Control: the same op with and without offsets differs only by the offset text.
    let plain = {
        // A transposed operand makes this cell Strided without any offset.
        let o = OpDef::elementwise("off", 2, &[ElementKind::F32], input(0) + input(1));
        let d = vec![
            desc(ElementKind::F32, &[4, 3], &[1, 4]),
            desc(ElementKind::F32, &[4, 3], &[3, 1]),
            desc(ElementKind::F32, &[4, 3], &[3, 1]),
        ];
        let key = structure_key(OpCategory::BinaryElementwise, &d, gfx906());
        let plan = try_build_plan(&o, &key).expect("plan");
        assert_eq!(plan.schedule, Schedule::Strided);
        Hip.lower(&plan).unwrap().source
    };
    assert!(plain.contains(" void rockfish_hip_off_f32(\n"), "{plain}");
    assert!(!plain.contains("off0") && !plain.contains("offo") && !plain.contains("+= off"));
}

#[test]
fn halves_take_the_same_arguments_and_bumps() {
    let got = lower(
        ElementKind::F16,
        input(0) + input(1),
        [DENSE, DENSE],
        Some((
            vec![BaseOffset::Runtime, BaseOffset::Zero],
            BaseOffset::Runtime,
        )),
    )
    .unwrap();
    assert!(got.contains("_off0o("), "{got}");
    assert!(
        got.contains("    long long off0,\n    long long offo,\n    long long n)"),
        "{got}"
    );
    assert!(
        got.contains("    in0 += off0;\n    out += offo;\n"),
        "{got}"
    );
}
