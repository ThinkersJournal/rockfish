//! Tests for the strided elementwise schedule of [`rockfish_hip_emit::Hip`].
//!
//! **What these show.** The emitted text is what is reviewed here: the unravel, the per-operand
//! offsets (a transposed operand, a broadcast axis, a fully broadcast operand), the `Coord` leaf
//! and the launch ABI, plus a typed refusal with a control that lowers for everything outside the
//! cell. **What they do not show:** that the text compiles under a HIP compiler or addresses the
//! right elements on a GPU (rig rows, `docs/DESIGN-R7.md` section 6).

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{Expr, OpDef, coord, input};
use unpopped::plan::{Schedule, try_build_plan};
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

fn gfx906() -> TargetId {
    TargetId::parse("rocm:gfx906").unwrap()
}

fn desc(dt: ElementKind, shape: &[i64], strides: &[i64]) -> OperandDesc {
    let esz = if dt == ElementKind::F64 { 8 } else { 4 };
    OperandDesc::new(shape.len(), shape, strides, dt, esz * 64)
}

/// Lower an elementwise op over `ins` (each `(shape, strides)`) into a dense `[4, 3]` output.
fn lower(dt: ElementKind, body: Expr, ins: &[(&[i64], &[i64])]) -> Result<String, LowerError> {
    let o = OpDef::elementwise("st", ins.len() as u8, &[dt], body);
    let mut d: Vec<OperandDesc> = ins.iter().map(|(s, st)| desc(dt, s, st)).collect();
    d.push(desc(dt, &[4, 3], &[3, 1]));
    let cat = match ins.len() {
        1 => OpCategory::UnaryElementwise,
        _ => OpCategory::BinaryElementwise,
    };
    let key = structure_key(cat, &d, gfx906());
    let plan = try_build_plan(&o, &key).expect("plan");
    assert_eq!(
        plan.schedule,
        Schedule::Strided,
        "test cell must be strided"
    );
    Hip.lower(&plan).map(|k| k.source)
}

const TRANSPOSED: (&[i64], &[i64]) = (&[4, 3], &[1, 4]);
const DENSE: (&[i64], &[i64]) = (&[4, 3], &[3, 1]);
const ROW_BCAST: (&[i64], &[i64]) = (&[4, 3], &[0, 1]);
const ALL_BCAST: (&[i64], &[i64]) = (&[4, 3], &[0, 0]);

/// Reviewed by reading. If the emitter changes this text, read the diff; do not regenerate it.
#[test]
fn golden_transposed_plus_row_broadcast() {
    let got = lower(
        ElementKind::F32,
        input(0) + input(1),
        &[TRANSPOSED, ROW_BCAST],
    )
    .unwrap();
    let body = got.split("extern \"C\"").nth(1).expect("a kernel");
    let want = " __global__ void rockfish_hip_st_f32(\n\
        \x20   const float* in0,\n\
        \x20   const float* in1,\n\
        \x20   float* out,\n\
        \x20   long long shape0,\n\
        \x20   long long shape1,\n\
        \x20   long long s0_0,\n\
        \x20   long long s0_1,\n\
        \x20   long long s1_0,\n\
        \x20   long long s1_1,\n\
        \x20   long long so_0,\n\
        \x20   long long so_1,\n\
        \x20   long long n)\n\
        {\n\
        \x20   #pragma clang fp contract(off)\n\
        \x20   for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;\n\
        \x20        i += (long long)gridDim.x * blockDim.x) {\n\
        \x20       long long lin = i;\n\
        \x20       long long c1 = lin % shape1;\n\
        \x20       lin /= shape1;\n\
        \x20       long long c0 = lin % shape0;\n\
        \x20       lin /= shape0;\n\
        \x20       long long o0 = c0*s0_0 + c1*s0_1;\n\
        \x20       long long o1 = c1*s1_1;\n\
        \x20       long long oo = c0*so_0 + c1*so_1;\n\
        \x20       out[oo] = (in0[o0] + in1[o1]);\n\
        \x20   }\n\
        }\n";
    assert_eq!(body, want);
    assert!(got.contains("(strided)"), "{got}");
}

#[test]
fn a_fully_broadcast_operand_is_loaded_once_before_the_loop() {
    let got = lower(ElementKind::F64, input(0) * input(1), &[DENSE, ALL_BCAST]).unwrap();
    let hoist = got.find("const double h1 = in1[0];").expect("hoist");
    let loop_at = got.find("for (long long i").expect("loop");
    assert!(hoist < loop_at, "{got}");
    assert!(got.contains("(in0[o0] * h1)"), "{got}");
    assert!(
        !got.contains("long long o1"),
        "a hoisted operand has no offset: {got}"
    );
}

#[test]
fn a_coord_leaf_reads_the_unravelled_axis_at_the_compute_type() {
    let f32 = lower(ElementKind::F32, input(0) + coord(1), &[TRANSPOSED]).unwrap();
    assert!(f32.contains("((float)c1)"), "{f32}");
    let f64 = lower(ElementKind::F64, input(0) + coord(0), &[TRANSPOSED]).unwrap();
    assert!(f64.contains("((double)c0)"), "{f64}");
}

#[test]
fn refusals_are_typed_and_each_has_a_control_that_lowers() {
    // Control.
    assert!(lower(ElementKind::F32, input(0) + coord(1), &[TRANSPOSED]).is_ok());
    // An integer strided cell is not spelled.
    assert!(matches!(
        lower(ElementKind::I32, input(0), &[TRANSPOSED]),
        Err(LowerError::UnsupportedSchedule { .. })
    ));
}
