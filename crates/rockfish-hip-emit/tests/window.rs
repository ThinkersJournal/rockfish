// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for the sliding-window pool schedule of [`rockfish_hip_emit::Hip`].
//!
//! **What these show.** The emitted text is what is reviewed here: the tap arithmetic, the fold
//! and the finalisation, each op's shape, and a typed refusal (with a control that lowers) for
//! everything outside the emitted cell. **What they do not show:** that the text compiles under a
//! HIP compiler or computes the oracle's numbers on a GPU (rig rows, `docs/DESIGN-R7.md` section 6).

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{Expr, OpDef, ReduceOp, UnaryOp, input, konst, param, reduced};
use unpopped::plan::{Schedule, try_build_plan};
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

const IN: [i64; 3] = [2, 5, 4];

fn gfx906() -> TargetId {
    TargetId::parse("rocm:gfx906").unwrap()
}

fn strides(shape: &[i64]) -> Vec<i64> {
    vec![shape[1] * shape[2], shape[2], 1]
}

/// Window geometry: size 3, stride 2, dilation 1, pad 1 each side.
fn k_out(k_in: i64) -> i64 {
    (k_in + 2 - 3) / 2 + 1
}

fn lower_with(
    dt: ElementKind,
    rop: ReduceOp,
    axis: u8,
    include_pad: bool,
    pre: Expr,
    post: Expr,
) -> Result<String, LowerError> {
    let o = OpDef::window(
        "win",
        1,
        &[dt],
        rop,
        axis,
        3,
        2,
        1,
        1,
        1,
        include_pad,
        pre,
        post,
    );
    let mut out = IN.to_vec();
    out[usize::from(axis)] = k_out(IN[usize::from(axis)]);
    let esz = if dt == ElementKind::F64 { 8 } else { 4 };
    let d = [
        OperandDesc::new(3, &IN, &strides(&IN), dt, esz * 64),
        OperandDesc::new(3, &out, &strides(&out), dt, esz * 64),
    ];
    let key = structure_key(OpCategory::Pooling, &d, gfx906());
    let plan = try_build_plan(&o, &key).expect("the window plans");
    assert!(
        matches!(plan.schedule, Schedule::Window { .. }),
        "{:?}",
        plan.schedule
    );
    Hip.lower(&plan).map(|k| k.source)
}

fn lower(
    dt: ElementKind,
    rop: ReduceOp,
    axis: u8,
    include_pad: bool,
) -> Result<String, LowerError> {
    lower_with(dt, rop, axis, include_pad, input(0), reduced(0))
}

/// Reviewed by reading. If the emitter changes this text, read the diff; do not regenerate it.
#[test]
fn golden_max_pool_over_the_innermost_axis() {
    let got = lower(ElementKind::F32, ReduceOp::Max, 2, false).unwrap();
    let body = got.split("extern \"C\"").nth(1).expect("a kernel");
    let want = " __global__ void rockfish_hip_win_f32(\n\
        \x20   const float* in0,\n\
        \x20   float* out,\n\
        \x20   long long outer,\n\
        \x20   long long k_in,\n\
        \x20   long long k_out,\n\
        \x20   long long inner)\n\
        {\n\
        \x20   #pragma clang fp contract(off)\n\
        \x20   const long long total = outer * k_out * inner;\n\
        \x20   for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;\n\
        \x20        t += (long long)gridDim.x * blockDim.x) {\n\
        \x20       const long long ob = t / (k_out * inner);\n\
        \x20       const long long rem = t - ob * (k_out * inner);\n\
        \x20       const long long o = rem / inner;\n\
        \x20       const long long ib = rem - o * inner;\n\
        \x20       const long long base = ob * k_in * inner + ib;\n\
        \x20       double acc = 0.0;\n\
        \x20       long long cnt = 0;\n\
        \x20       bool have = false;\n\
        \x20       for (int kk = 0; kk < 3; ++kk) {\n\
        \x20           const long long p = o * 2 - 1 + (long long)kk * 1;\n\
        \x20           if (p < 0 || p >= k_in) { continue; }\n\
        \x20           double v = (double)(in0[base + p * inner]);\n\
        \x20           if (!have || v != v || v > acc) { acc = v; }\n\
        \x20           cnt += 1;\n\
        \x20           have = true;\n\
        \x20       }\n\
        \x20       (void)cnt;\n\
        \x20       (void)have;\n\
        \x20       out[t] = (float)(have ? acc : -__builtin_inf());\n\
        \x20   }\n\
        }\n";
    assert_eq!(body, want);
    assert!(got.contains("(window)"), "{got}");
}

#[test]
fn each_op_has_its_own_fold_and_finalisation() {
    let sum = lower(ElementKind::F32, ReduceOp::Sum, 2, true).unwrap();
    assert!(sum.contains("acc = acc + v;") && sum.contains("out[t] = (float)(acc);"));
    let prod = lower(ElementKind::F32, ReduceOp::Prod, 2, true).unwrap();
    assert!(prod.contains("double acc = 1.0;") && prod.contains("acc = acc * v;"));
    let min = lower(ElementKind::F64, ReduceOp::Min, 2, true).unwrap();
    assert!(min.contains("v < acc"));
    assert!(min.contains("have ? acc : __builtin_inf()") && min.contains("out[t] = (double)"));
    // Mean: the divisor policy is the whole difference between the two.
    let with_pad = lower(ElementKind::F32, ReduceOp::Mean, 2, true).unwrap();
    let without = lower(ElementKind::F32, ReduceOp::Mean, 2, false).unwrap();
    assert!(with_pad.contains("acc / (double)3"), "{with_pad}");
    assert!(
        without.contains("(cnt > 0) ? (acc / (double)cnt) : 0.0"),
        "{without}"
    );
    assert_ne!(with_pad, without);
}

/// The pooled axis can be any axis: the emitted text is the same (the `(outer, k, inner)` split
/// is a launch argument), and the geometry is read from the plan, not hard-wired.
#[test]
fn a_non_innermost_axis_lowers_to_the_same_kernel() {
    let inner = lower(ElementKind::F32, ReduceOp::Sum, 2, true).unwrap();
    let mid = lower(ElementKind::F32, ReduceOp::Sum, 1, true).unwrap();
    let strip = |s: &str| s.split("extern \"C\"").nth(1).unwrap().to_string();
    assert_eq!(strip(&inner), strip(&mid));
    assert_ne!(
        inner, mid,
        "the cell token differs, so the header comment does"
    );
}

#[test]
fn refusals_are_typed_and_each_has_a_control_that_lowers() {
    // Control.
    assert!(lower(ElementKind::F32, ReduceOp::Max, 2, false).is_ok());
    // An integer window is spelled (below); an integer average pool is refused by Unpopped's plan
    // gate before it reaches this emitter, so there is nothing to lower.
    assert!(lower(ElementKind::I32, ReduceOp::Sum, 2, false).is_ok());
    // A post-expression that reads an input operand (not only the folded value) is not spelled;
    // control: one over the folded value alone lowers.
    assert!(
        lower_with(
            ElementKind::F32,
            ReduceOp::Sum,
            2,
            true,
            input(0),
            reduced(0) + reduced(0)
        )
        .is_ok()
    );
    assert!(matches!(
        lower_with(
            ElementKind::F32,
            ReduceOp::Sum,
            2,
            true,
            input(0),
            reduced(0) + input(0)
        ),
        Err(LowerError::UnsupportedPlanShape { .. })
    ));
}

#[test]
fn a_post_expression_reads_the_folded_value_at_the_compute_type() {
    let f32k = lower_with(
        ElementKind::F32,
        ReduceOp::Mean,
        2,
        false,
        input(0),
        reduced(0) * konst(0.5),
    )
    .unwrap();
    assert!(
        f32k.contains("const float res = (float)((cnt > 0) ? (acc / (double)cnt) : 0.0);"),
        "{f32k}"
    );
    assert!(f32k.contains("out[t] = (float)("), "{f32k}");
    assert!(f32k.contains("(res *"), "{f32k}");
    // An integer window folds in the unsigned type and hands the epilogue the dtype.
    let i32k = lower_with(
        ElementKind::I32,
        ReduceOp::Sum,
        2,
        false,
        input(0),
        reduced(0) + reduced(0),
    )
    .unwrap();
    assert!(i32k.contains("const int res = (int)(acc);"), "{i32k}");
    // A halves window: the epilogue runs at float, then one rounding on store.
    let f16k = lower_with(
        ElementKind::F16,
        ReduceOp::Max,
        2,
        true,
        input(0),
        reduced(0) + reduced(0),
    )
    .unwrap();
    assert!(
        f16k.contains("const float res = (float)(have ? acc : -__builtin_inf());"),
        "{f16k}"
    );
    assert!(
        f16k.contains("out[t] = unpopped_f16_store((float)("),
        "{f16k}"
    );
}

#[test]
fn a_post_parameter_is_a_trailing_argument_except_at_a_half() {
    let k = lower_with(
        ElementKind::F32,
        ReduceOp::Sum,
        2,
        false,
        input(0),
        reduced(0) * param(0),
    )
    .unwrap();
    assert!(
        k.contains(
            "long long inner,
    float p0)"
        ),
        "{k}"
    );
    assert!(
        lower_with(
            ElementKind::F16,
            ReduceOp::Sum,
            2,
            false,
            input(0),
            reduced(0) * param(0)
        )
        .is_err()
    );
}

/// The pre-map is lowered over the loaded tap, so a body the elementwise cell spells works here.
#[test]
fn a_pre_map_is_lowered_over_each_tap() {
    let got = lower_with(
        ElementKind::F32,
        ReduceOp::Sum,
        2,
        true,
        input(0).unary(UnaryOp::Abs),
        reduced(0),
    )
    .unwrap();
    assert!(got.contains("fabsf(in0[base + p * inner])"), "{got}");
}

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

/// An integer window folds in the dtype itself: no `double`, so no rounding above 2^53 and no
/// float NaN test; Sum/Prod wrap in the unsigned type of the same width (signed overflow is
/// undefined in C++), Max/Min compare in the dtype.
#[test]
fn integer_windows_fold_exactly_at_every_integer_dtype() {
    for (dt, tag, t, u) in INTS {
        let kernel = |op| {
            let src = lower(dt, op, 2, false).unwrap_or_else(|e| panic!("{tag} {op:?}: {e:?}"));
            src.split("extern \"C\"").nth(1).unwrap().to_string()
        };
        let sum = kernel(ReduceOp::Sum);
        assert!(sum.contains(&format!("{u} acc = ({u})0;")), "{tag}: {sum}");
        assert!(
            sum.contains(&format!("acc = acc + ({u})v;")),
            "{tag}: {sum}"
        );
        assert!(
            sum.contains(&format!("{t} v = ({t})(in0[base + p * inner]);")),
            "{tag}: {sum}"
        );
        assert!(
            sum.contains(&format!("out[t] = ({t})(acc);")),
            "{tag}: {sum}"
        );
        let prod = kernel(ReduceOp::Prod);
        assert!(
            prod.contains(&format!("{u} acc = ({u})1;")),
            "{tag}: {prod}"
        );
        assert!(
            prod.contains(&format!("acc = acc * ({u})v;")),
            "{tag}: {prod}"
        );
        let max = kernel(ReduceOp::Max);
        assert!(max.contains(&format!("{t} acc = 0;")), "{tag}: {max}");
        assert!(
            max.contains("if (!have || v > acc) { acc = v; }"),
            "{tag}: {max}"
        );
        let min = kernel(ReduceOp::Min);
        assert!(
            min.contains("if (!have || v < acc) { acc = v; }"),
            "{tag}: {min}"
        );
        for (op, src) in [("Sum", &sum), ("Prod", &prod), ("Max", &max), ("Min", &min)] {
            assert!(
                !src.contains("double"),
                "{tag} {op}: a double leaked in: {src}"
            );
            assert!(
                !src.contains("v != v"),
                "{tag} {op}: a NaN test leaked in: {src}"
            );
            assert!(
                !src.contains("__builtin_inf"),
                "{tag} {op}: an inf leaked in: {src}"
            );
        }
    }
}

/// Control for the test above: the float cell is unchanged (it still folds in `double`), so the
/// integer assertions fail if the integer branch stops being taken.
#[test]
fn the_float_window_still_folds_in_double() {
    let f = lower(ElementKind::F32, ReduceOp::Sum, 2, false).unwrap();
    assert!(f.contains("double acc = 0.0;") && f.contains("(double)(in0[base + p * inner])"));
}

#[test]
fn a_half_window_widens_on_load_folds_in_double_and_rounds_on_store() {
    for (dt, tag, ld, st) in [
        (
            ElementKind::F16,
            "f16",
            "unpopped_f16_load",
            "unpopped_f16_store",
        ),
        (
            ElementKind::Bf16,
            "bf16",
            "unpopped_bf16_load",
            "unpopped_bf16_store",
        ),
    ] {
        let mean = lower(dt, ReduceOp::Mean, 1, false).unwrap();
        assert!(mean.contains(&format!("rockfish_hip_win_{tag}(")), "{mean}");
        assert!(mean.contains("const unsigned short* in0,"), "{mean}");
        assert!(mean.contains("    unsigned short* out,"), "{mean}");
        assert!(
            mean.contains(&format!(
                "double v = (double)({ld}(in0[base + p * inner]));"
            )),
            "{mean}"
        );
        assert!(mean.contains("double acc = 0.0;"), "{mean}");
        assert!(
            mean.contains(&format!(
                "out[t] = {st}((float)((cnt > 0) ? (acc / (double)cnt) : 0.0));"
            )),
            "{mean}"
        );
        let max = lower(dt, ReduceOp::Max, 2, true).unwrap();
        assert!(
            max.contains(&format!(
                "out[t] = {st}((float)(have ? acc : -__builtin_inf()));"
            )),
            "{max}"
        );
    }
}
