//! Golden vectors and grammar corpus for [`rockfish_hip_parse::HIP`].
//!
//! Every kernel here is hand-written for this repository (none copied from AMD or third-party
//! samples). **Not a correctness proof for HIP parsing in general**: these are the idioms below,
//! checked against `tree-sitter-cuda` 0.21; nothing is compiled or run on a GPU.

use rockfish_hip_parse::{HIP, HIP_RESIDUE, parse_hip};
use unpopped::convert::{lift, lift_elementwise, lift_reduction, lift_scan};
use unpopped::ir::{Access, BinaryOp, ReduceOp, ScalarExpr, UnaryOp};
use unpopped::lift::LiftError;
use unpopped_vocab::ElementKind;

const F32: &[ElementKind] = &[ElementKind::F32];

fn body(src: &str) -> ScalarExpr {
    lift_elementwise(&HIP, src, "x", F32).unwrap().op.body
}

fn input(k: u8) -> Box<ScalarExpr> {
    Box::new(ScalarExpr::Input(k))
}

/// HIP-specific syntax, each as a complete translation unit; the flag is the **measured** answer
/// to "does `tree-sitter-cuda` 0.21 parse it with no `ERROR`/missing node". A construct listed
/// `false` is a known gap: the lift may still work (the walkers look only at the store), but the
/// tree is not clean, so no claim is made for it.
const CORPUS: &[(&str, &str, bool)] = &[
    (
        "cuda-style indices",
        "__global__ void k(const float* in0, float* out, long long n) {\n\
         long long i = blockIdx.x * blockDim.x + threadIdx.x;\n\
         if (i < n) { out[i] = in0[i] + 1.0f; }\n}",
        true,
    ),
    (
        "hip macro indices",
        "__global__ void k(const float* in0, float* out, long long n) {\n\
         long long i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n\
         if (i < n) { out[i] = in0[i] + 1.0f; }\n}",
        true,
    ),
    (
        "launch bounds after global",
        "__global__ __launch_bounds__(256) void k(const float* in0, float* out, long long n) {\n\
         out[0] = in0[0];\n}",
        true,
    ),
    (
        "restrict qualifiers",
        "__global__ void k(const float* __restrict__ in0, float* __restrict__ out) {\n\
         out[0] = in0[0];\n}",
        true,
    ),
    (
        "extern C",
        "extern \"C\" __global__ void k(const float* in0, float* out) { out[0] = in0[0]; }",
        true,
    ),
    (
        "device helper",
        "__device__ __forceinline__ float relu(float x) { return fmaxf(x, 0.0f); }\n\
         __global__ void k(const float* in0, float* out) { out[0] = relu(in0[0]); }",
        true,
    ),
    (
        "include directive",
        "#include <hip/hip_runtime.h>\n\
         __global__ void k(const float* in0, float* out) { out[0] = in0[0]; }",
        true,
    ),
    (
        "triple chevron launch",
        "__global__ void k(float* out) { out[0] = 0.0f; }\n\
         void host(float* d) { k<<<1, 64>>>(d); }",
        true,
    ),
    (
        "hipLaunchKernelGGL host call",
        "__global__ void k(float* out) { out[0] = 0.0f; }\n\
         void host(float* d) { hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, d); }",
        true,
    ),
    (
        "grid-stride loop",
        "__global__ void k(const float* in0, float* out, long long n) {\n\
         for (long long i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < n;\n\
              i += hipGridDim_x * hipBlockDim_x) { out[i] = in0[i]; }\n}",
        true,
    ),
    (
        "hip fp16 type",
        "__global__ void k(const __half* in0, __half* out) { out[0] = in0[0]; }",
        true,
    ),
];

#[test]
fn corpus_parses_as_recorded() {
    for (name, src, clean) in CORPUS {
        let tree = parse_hip(src).unwrap_or_else(|| panic!("{name}: no tree"));
        let has_error = tree.root_node().has_error();
        assert_eq!(!has_error, *clean, "{name}: has_error={has_error}");
    }
}

/// Control for the corpus test: the same check must be able to fail. Text that is not C++ at all
/// has to report an error node, or `corpus_parses_as_recorded` proves nothing.
#[test]
fn corpus_control_a_broken_kernel_reports_an_error() {
    let tree = parse_hip("__global__ void k( { out[ = ;;; )))").unwrap();
    assert!(tree.root_node().has_error());
}

#[test]
fn lifts_fused_multiply_add_with_hip_indices() {
    let src = "__global__ void mul(const float* in0, const float* in1, const float* in2, float* out, long long n) {\n\
        long long i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n\
        for (; i < n; i += hipGridDim_x * hipBlockDim_x) { out[i] = in0[i] * in1[i] + in2[i]; }\n}";
    let lifted = lift_elementwise(&HIP, src, "fma", F32).unwrap();
    assert_eq!(lifted.n_inputs, 3);
    assert_eq!(
        lifted.op.body,
        ScalarExpr::Add(Box::new(ScalarExpr::Mul(input(0), input(1))), input(2))
    );
}

#[test]
fn lifts_through_launch_bounds_and_restrict() {
    let src = "__global__ __launch_bounds__(256) void k(const float* __restrict__ in0, float* __restrict__ out) { out[i] = expf(in0[i]); }";
    assert_eq!(body(src), ScalarExpr::Unary(UnaryOp::Exp, input(0)));
}

#[test]
fn precedence_comes_from_the_grammar() {
    let src = "__global__ void k(const float* in0, const float* in1, const float* in2, float* out){ out[i] = in0[i] + in1[i]*in2[i]; }";
    assert_eq!(
        body(src),
        ScalarExpr::Add(input(0), Box::new(ScalarExpr::Mul(input(1), input(2))))
    );
}

#[test]
fn fmaxf_is_ieee_max() {
    let src = "__global__ void k(const float* in0, float* out){ out[i] = fmaxf(in0[i], 0.0f); }";
    assert_eq!(
        body(src),
        ScalarExpr::Binary(
            BinaryOp::FmaxIeee,
            input(0),
            Box::new(ScalarExpr::Const(0.0))
        )
    );
}

/// Fast-math spellings are approximations; lifting them to the exact op would turn an
/// approximation into the accurate function on re-emit. They must refuse.
#[test]
fn fast_math_intrinsics_refuse_not_lift() {
    for f in ["__expf", "__logf", "__sinf", "__cosf"] {
        let src =
            format!("__global__ void k(const float* in0, float* out){{ out[i] = {f}(in0[i]); }}");
        assert!(
            matches!(
                lift_elementwise(&HIP, &src, "x", F32),
                Err(LiftError::Unrecognized(_))
            ),
            "{f}: {:?}",
            lift_elementwise(&HIP, &src, "x", F32)
        );
    }
}

/// Positive control for the line above: the accurate spelling of the same math does lift.
#[test]
fn accurate_math_spelling_lifts() {
    let src = "__global__ void k(const float* in0, float* out){ out[i] = expf(in0[i]); }";
    assert!(lift_elementwise(&HIP, src, "x", F32).is_ok());
}

#[test]
fn neighbour_access_is_unrecognized_not_inexpressible() {
    let src = "__global__ void k(const float* in0, float* out){ out[i] = in0[i+1]; }";
    assert!(matches!(
        lift_elementwise(&HIP, src, "x", F32),
        Err(LiftError::Unrecognized(_))
    ));
}

#[test]
fn non_kernel_is_not_a_kernel() {
    assert!(matches!(
        lift_elementwise(&HIP, "void f(float* out){ out[0] = 1.0f; }", "x", F32),
        Err(LiftError::NotAKernel)
    ));
}

/// One positive and one negative per residue entry. The positive embeds the marker in an otherwise
/// liftable kernel and must be refused as `Inexpressible(marker)`; the negative is the same kernel
/// without it and must lift (so the refusal is caused by the marker, not by the kernel).
#[test]
fn each_residue_entry_refuses_and_its_absence_lifts() {
    let clean = "__global__ void k(const float* in0, float* out){ out[i] = in0[i] + 1.0f; }";
    assert!(
        lift_elementwise(&HIP, clean, "x", F32).is_ok(),
        "control kernel must lift"
    );
    for marker in HIP_RESIDUE {
        let dirty = format!(
            "__global__ void k(const float* in0, float* out){{ /* {marker} */ out[i] = in0[i] + 1.0f; }}"
        );
        match lift_elementwise(&HIP, &dirty, "x", F32) {
            Err(LiftError::Inexpressible(w)) => {
                assert!(
                    HIP_RESIDUE.contains(&w.as_str()) && dirty.contains(w.as_str()),
                    "{marker}: refused for {w}"
                );
            }
            other => panic!("{marker}: expected Inexpressible, got {other:?}"),
        }
    }
}

/// Realistic uses of the headline residue constructs, not just a comment holding the word.
#[test]
fn realistic_residue_kernels_refuse() {
    let cases: &[(&str, &str)] = &[
        (
            "__global__ void k(float* out){ __shared__ float s[64]; s[0] = 1.0f; __syncthreads(); out[i] = s[0]; }",
            "__shared__",
        ),
        (
            "__global__ void k(const float* in0, float* out){ atomicAdd(&out[0], in0[i]); }",
            "atomic",
        ),
        (
            "__global__ void k(const float* in0, float* out){ out[i] = __shfl_down(in0[i], 1); }",
            "__shfl",
        ),
        (
            "__global__ void k(float* out){ out[i] = __builtin_amdgcn_workitem_id_x(); }",
            "__builtin_amdgcn",
        ),
    ];
    for (src, want) in cases {
        assert!(
            matches!(lift_elementwise(&HIP, src, "x", F32), Err(LiftError::Inexpressible(ref w)) if w == want
                || (*want == "__shared__" && w == "__shared__")),
            "{want}: {:?}",
            lift_elementwise(&HIP, src, "x", F32)
        );
    }
}

#[test]
fn lifts_sum_reduction() {
    let src = "__global__ void sum(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc += in0[i]; out[0] = acc; }";
    let l = lift_reduction(&HIP, src, "sum", F32).unwrap();
    assert_eq!(l.n_inputs, 1);
    assert_eq!(l.op.body, ScalarExpr::Input(0));
    assert!(matches!(
        l.op.access,
        Access::Reduction {
            op: ReduceOp::Sum,
            ..
        }
    ));
}

#[test]
fn lifts_cumsum_and_does_not_misclassify_it() {
    let src = "__global__ void cs(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++){ acc += in0[i]; out[i] = acc; } }";
    let l = lift_scan(&HIP, src, "cs", F32).unwrap();
    assert!(matches!(
        l.op.access,
        Access::Scan {
            op: ReduceOp::Sum,
            reverse: false,
            exclusive: false,
            ..
        }
    ));
    // Unpopped measured a cumsum silently accepted as elementwise once; `lift` must keep it a scan.
    let any = lift(&HIP, src, "cs", F32).unwrap();
    assert!(
        matches!(any.op.access, Access::Scan { .. }),
        "got {:?}",
        any.op.access
    );
}
