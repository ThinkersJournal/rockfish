//! The HIP C++ [`unpopped::backend::Backend`]: the emit side (IR -> HIP source text) of
//! Rockfish's HIP C++ ownership (see `docs/DESIGN-R7.md`).
//!
//! **Text only.** [`Hip`] produces source; nothing here has been through a HIP compiler
//! (`hipcc`, hipRTC) and nothing was run on a GPU. Every spelling below is a claim about what
//! a HIP compiler accepts and what it computes, labelled UNVERIFIED until the rig compiles the
//! golden kernels (a rig row in `docs/DESIGN-R7.md` section 6). The emitted text deliberately
//! carries no status comment: a claim written into an artifact is invalidated by verifying it.
//!
//! # Scope of this slice (R7b)
//!
//! One shape only: the scalar contiguous elementwise cell ([`Schedule::Scalar`]), one output,
//! `f32`, on a `rocm:` target. Everything else is a typed [`LowerError`], never source: the
//! rule from `Backend::lower` is *refuse rather than approximate*.
//!
//! # Every spelling is stated here
//!
//! The emitter does not call `unpopped::cfamily`'s shared spellers and does not rewrite CUDA
//! output. Each operation this crate spells appears below as a literal match arm; an operation
//! with no arm is declined (see `unary` and `binary`). The set is exactly the inverse of the
//! call names `rockfish-hip-parse` lifts, which is what makes the round-trip tests meaningful.
//!
//! # Wavefront size
//!
//! `KernelPlan` carries no subgroup / wavefront size (read in Unpopped 0.38.4: the only
//! `warp` mentions in `plan.rs` are doc comments on the scan and reduce schedules). Elementwise
//! needs none. The reduction and scan slices must take the width from the target or from the
//! device-side `warpSize`, never assume 32.

use unpopped::backend::{
    Backend, Decline, DeclinedOp, DtypeSupport, GeneratedKernel, LowerError, Lowering, Spelling,
    lower_dag,
};
use unpopped::ir::{Access, ArithOp, BaseOffset, BinaryOp, DagNode, ExprDag, UnaryOp, WriteIndex};
use unpopped::plan::{KernelPlan, Schedule};
use unpopped_vocab::{ElementKind, TargetId};

/// The HIP C++ backend. Lowers a [`KernelPlan`] to HIP kernel source text.
#[derive(Copy, Clone, Debug, Default)]
pub struct Hip;

/// The `TargetId` namespace this backend serves. `rocm:gfx942` is the spelling Unpopped's own
/// derivation vectors use for an AMD target.
const NAMESPACE: &str = "rocm";

impl Backend for Hip {
    fn name(&self) -> &str {
        "hip"
    }

    fn provider(&self) -> &str {
        "rockfish"
    }

    /// `f32` on a `rocm:` target is the only `Supported` answer.
    ///
    /// `float` is a type of the language, so this is a statement about the emitter's scope and
    /// not about any one gfx part's instruction set. Every other dtype answers `Unsupported`
    /// (the safe answer when unsure, per `Backend::supports_dtype`) until its HIP type name,
    /// header and per-architecture availability are read from AMD's headers or measured: the
    /// vendored `vendor/hip/include/hip` has neither `hip_fp16.h` nor `hip_bf16.h`. A target
    /// outside the `rocm:` namespace is `Unsupported` too: this text could never compile there.
    fn supports_dtype(&self, dtype: ElementKind, target: TargetId) -> DtypeSupport {
        if target.namespace() != NAMESPACE {
            return DtypeSupport::Unsupported;
        }
        match dtype {
            ElementKind::F32 => DtypeSupport::Supported,
            _ => DtypeSupport::Unsupported,
        }
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        if !self
            .supports_dtype(plan.dtype, plan.key.target)
            .is_supported()
        {
            return Err(LowerError::UnsupportedDtype {
                dtype: plan.dtype,
                detail: format!(
                    "hip backend: only f32 on a `{NAMESPACE}:` target is spelled (target `{}`); \
                     see `Hip::supports_dtype`",
                    plan.key.target
                ),
            });
        }
        if plan.out_dtype != plan.dtype {
            return Err(shape(format!(
                "hip backend: output dtype {:?} differs from compute dtype {:?}",
                plan.out_dtype, plan.dtype
            )));
        }
        check_plan_shape(plan)?;
        match plan.schedule {
            Schedule::Scalar => emit_scalar(plan),
            other => Err(LowerError::UnsupportedSchedule {
                detail: format!(
                    "hip backend: the scalar contiguous elementwise schedule only, got {other:?}"
                ),
            }),
        }
    }
}

fn shape(detail: String) -> LowerError {
    LowerError::UnsupportedPlanShape { detail }
}

/// Everything about the plan that is not the body or the schedule. The scalar emitter writes
/// `out[i] = body` over contiguous operands; any plan feature that would change addressing,
/// the number of stores or the launch arguments is outside it, and is refused here instead of
/// being silently ignored.
fn check_plan_shape(plan: &KernelPlan<'_>) -> Result<(), LowerError> {
    if !is_c_identifier_tail(plan.op_name) {
        return Err(shape(format!(
            "hip backend: op name {:?} is not [A-Za-z0-9_]+, so it cannot form a kernel symbol \
             or sit in a comment safely",
            plan.op_name
        )));
    }
    if plan.n_outputs != 1 || !plan.extra_out_bodies.is_empty() || !plan.extra_out_dtypes.is_empty()
    {
        return Err(shape(format!(
            "hip backend: single output only; op '{}' has {}",
            plan.op_name, plan.n_outputs
        )));
    }
    if !matches!(plan.access, Access::Elementwise) {
        return Err(shape(format!(
            "hip backend: elementwise access only; op '{}' has {:?}",
            plan.op_name, plan.access
        )));
    }
    if !plan.views.is_empty()
        || !plan.read_index.is_empty()
        || !matches!(plan.write_index, WriteIndex::Direct)
        || !plan.base_offsets.is_empty()
        || !matches!(plan.out_base_offset, BaseOffset::Zero)
    {
        return Err(shape(format!(
            "hip backend: op '{}' has a view, gather, scatter or base offset, which the scalar \
             contiguous emitter does not address",
            plan.op_name
        )));
    }
    Ok(())
}

fn is_c_identifier_tail(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// The scalar contiguous elementwise kernel: one grid-stride loop over `n` elements.
///
/// The loop is a grid-stride loop so the result does not depend on the launch configuration
/// covering `n` exactly. The first index is widened before the multiply (`blockIdx.x *
/// blockDim.x` is a 32-bit product otherwise). No `__restrict__`: that would assert that
/// `out` does not alias an input, which this emitter has no basis to claim.
///
/// `#pragma clang fp contract(off)`: clang's default for HIP is, to the best of the author's
/// reading, `-ffp-contract=fast`, which may fuse `a * b + c` into one rounding. The IR's
/// `Add(Mul(a, b), c)` is two roundings (an `Fma` node is the one-rounding form, and is
/// declined here), so the contraction is switched off in the kernel body. UNVERIFIED against
/// a HIP compiler.
fn emit_scalar(plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
    let dag = ExprDag::from_expr(plan.body);
    if (0..dag.len() as u32).any(|i| matches!(dag.node(i), DagNode::Param(_))) {
        return Err(shape(format!(
            "hip backend: op '{}' reads a runtime scalar parameter; launch parameters are not \
             emitted yet",
            plan.op_name
        )));
    }
    let n_inputs = plan.n_inputs;
    let leaf = |idx: u8| {
        Ok(if idx < n_inputs {
            Spelling::Spelled(format!("in{idx}[i]"))
        } else {
            Spelling::Declined(Decline::UnsupportedOp {
                op: DeclinedOp::Leaf(idx),
                why: format!("input {idx} is past the plan's {n_inputs} input operands"),
            })
        })
    };
    let un = |op: UnaryOp, x: String| unary(op, &x);
    let bin = |op: BinaryOp, a: String, b: String| binary(op, &a, &b);
    let ar = |op: ArithOp, a: String, b: String| arith(op, &a, &b);
    let cst = |v: f64| constant(v);
    let no_reduced = |i: u8| {
        Ok(Spelling::Declined(Decline::UnsupportedOp {
            op: DeclinedOp::Reduced(i),
            why: "a reduced-scalar leaf has no place in an elementwise kernel".to_string(),
        }))
    };
    let no_coord = |i: u8| {
        Ok(Spelling::Declined(Decline::UnsupportedOp {
            op: DeclinedOp::Coord(i),
            why: "output coordinates need the strided schedule, which this backend does not emit"
                .to_string(),
        }))
    };
    // Seams not set here (select, fma, cast, bitcast, qualified, integer ops, locals) keep the
    // builder's declining defaults: those nodes are refused, not approximated.
    let lowering = Lowering::builder(&leaf, &un, &bin)
        .arith(&ar)
        .constant(&cst)
        .reduced(&no_reduced)
        .coord(&no_coord)
        .compute_dtype(ElementKind::F32)
        .build();
    let (prelude, root) = lower_dag(&dag, "float", &lowering)?;

    let name = format!("rockfish_hip_{}_f32", plan.op_name);
    let mut s = String::new();
    s.push_str(&format!(
        "// Generated by rockfish-hip-emit (hip backend) - do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    ));
    s.push_str("#include <hip/hip_runtime.h>\n\n");
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    for i in 0..n_inputs {
        s.push_str(&format!("    const float* in{i},\n"));
    }
    s.push_str("    float* out,\n    long long n)\n{\n");
    s.push_str("    #pragma clang fp contract(off)\n");
    s.push_str(
        "    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;\n         \
         i += (long long)gridDim.x * blockDim.x) {\n",
    );
    for decl in &prelude {
        s.push_str(&format!("        {decl}\n"));
    }
    s.push_str(&format!("        out[i] = {root};\n    }}\n}}\n"));
    Ok(GeneratedKernel::new(name, s))
}

fn declined_op(op: DeclinedOp, why: &str) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op,
        why: why.to_string(),
    }))
}

/// The unary operations spelled, as `f32` C library calls available in HIP device code.
/// A [`UnaryOp`] with no arm is declined. Notable absences, each because the IR's definition
/// and the C library function are not obviously the same function (or no spelling has been
/// decided): `Round` (ties-to-even vs C `roundf`, ties away), `Sign`, `Relu`/`Step`/`Sigmoid`/
/// `Gelu`/`Silu` (compositions), `Rsqrt`/`Recip`/`Sqr` (an accuracy or form decision),
/// `Erf` and the hyperbolic/inverse families (library availability unchecked).
fn unary(op: UnaryOp, x: &str) -> Result<Spelling, LowerError> {
    let f = |name: &str| Ok(Spelling::Spelled(format!("{name}({x})")));
    match op {
        UnaryOp::Neg => Ok(Spelling::Spelled(format!("(-{x})"))),
        UnaryOp::Abs => f("fabsf"),
        UnaryOp::Sqrt => f("sqrtf"),
        UnaryOp::Exp => f("expf"),
        UnaryOp::Log => f("logf"),
        UnaryOp::Tanh => f("tanhf"),
        UnaryOp::Sin => f("sinf"),
        UnaryOp::Cos => f("cosf"),
        UnaryOp::Floor => f("floorf"),
        UnaryOp::Ceil => f("ceilf"),
        UnaryOp::Trunc => f("truncf"),
        UnaryOp::Exp2 => f("exp2f"),
        other => declined_op(
            DeclinedOp::Unary(other),
            "no HIP spelling decided for this unary op in rockfish-hip-emit",
        ),
    }
}

/// The binary operations spelled. `FmaxIeee`/`FminIeee` are C `fmaxf`/`fminf` (NaN-suppressing);
/// the NaN-propagating `Max`/`Min` are different functions and are declined, as are the
/// comparisons, remainders, `Nextafter`, and every integer, bitwise and logical op.
fn binary(op: BinaryOp, a: &str, b: &str) -> Result<Spelling, LowerError> {
    let f = |name: &str| Ok(Spelling::Spelled(format!("{name}({a}, {b})")));
    match op {
        BinaryOp::FmaxIeee => f("fmaxf"),
        BinaryOp::FminIeee => f("fminf"),
        BinaryOp::Pow => f("powf"),
        BinaryOp::Atan2 => f("atan2f"),
        BinaryOp::Copysign => f("copysignf"),
        other => declined_op(
            DeclinedOp::Binary(other),
            "no HIP spelling decided for this binary op in rockfish-hip-emit",
        ),
    }
}

fn arith(op: ArithOp, a: &str, b: &str) -> Result<Spelling, LowerError> {
    let sym = match op {
        ArithOp::Add => "+",
        ArithOp::Sub => "-",
        ArithOp::Mul => "*",
        ArithOp::Div => "/",
        other => {
            return declined_op(
                DeclinedOp::Arith(other),
                "no HIP spelling for this arithmetic node",
            );
        }
    };
    Ok(Spelling::Spelled(format!("({a} {sym} {b})")))
}

/// An `f32` literal for a body constant.
///
/// The IR defines a constant at an `f32` compute dtype as rounded to `f32`, round to nearest
/// even (`unpopped::optimize::Compute::F32`: "the bits the device would hold"), and
/// `generate` does not run the optimizer, so an authored `0.1` arrives here as the `f64`
/// `0.1`. Spelling `(v as f32)` with an `f` suffix is that same rounding done once, here, and
/// the printed digits are the shortest that name that exact `f32`, so the compiler's own
/// decimal-to-float conversion lands on the same bits.
///
/// Refused rather than approximated: a non-finite value (`NAN`/`INFINITY` are macros whose HIP
/// availability is unchecked), a finite value that overflows `f32` to infinity, and a nonzero
/// value that underflows to zero. Those change the value's class, which rounding does not.
fn constant(v: f64) -> Result<Spelling, LowerError> {
    let narrowed = v as f32;
    if !v.is_finite() || !narrowed.is_finite() || (v != 0.0 && narrowed == 0.0) {
        return declined_op(
            DeclinedOp::Constant,
            "constant is non-finite, overflows f32, or underflows f32 to zero",
        );
    }
    let lit = format!("{narrowed:?}f");
    // Debug of an f32 is always a float literal body ("1.0", "0.1", "1e-7"); a negative value
    // is parenthesised so it can follow any operator.
    Ok(Spelling::Spelled(if narrowed.is_sign_negative() {
        format!("({lit})")
    } else {
        lit
    }))
}
