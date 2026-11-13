//! `f16` and `bf16` in `rockfish-hip-emit`: stored as `unsigned short`, computed at `float`
//! through Unpopped's software codec (`unpopped::cfamily::half_helpers`) in HIP's spelling.
//!
//! **What these show:** the text produced, and what stays refused. **What they do not show:**
//! that a HIP compiler accepts it (rig row). The text, with a shim `hip_runtime.h`, was compiled
//! by a host C++ compiler as a one-off (see the CHANGELOG); that is not repeated here.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, DtypeSupport, LowerError};
use unpopped::ir::{BinaryOp, OpDef, input, param};
use unpopped::plan::{Schedule, try_build_plan_for};
use unpopped::try_generate;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key};

const HALVES: [(ElementKind, &str); 2] = [(ElementKind::F16, "f16"), (ElementKind::Bf16, "bf16")];

fn key(n_in: usize, dt: ElementKind, target: &str, align: u32) -> StructureKey {
    let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
    let cat = if n_in == 1 {
        OpCategory::UnaryElementwise
    } else {
        OpCategory::BinaryElementwise
    };
    structure_key(
        cat,
        &vec![a; n_in + 1],
        TargetId::parse(target).expect("target"),
    )
}

fn add(dt: ElementKind) -> OpDef {
    OpDef::elementwise("add", 2, &[dt], input(0) + input(1))
}

#[test]
fn halves_are_supported_on_rocm_only() {
    for (dt, _) in HALVES {
        let rocm = TargetId::parse("rocm:gfx906").unwrap();
        assert_eq!(Hip.supports_dtype(dt, rocm), DtypeSupport::Supported);
        let cuda = TargetId::parse("cuda:sm_80").unwrap();
        assert_eq!(Hip.supports_dtype(dt, cuda), DtypeSupport::Unsupported);
    }
}

#[test]
fn a_half_kernel_stores_16_bits_and_computes_at_float() {
    for (dt, tag) in HALVES {
        let o = add(dt);
        let s = try_generate(&o, &key(2, dt, "rocm:gfx906", 1), &Hip)
            .unwrap_or_else(|e| panic!("{dt:?}: {e:?}"))
            .source;
        assert!(s.contains(&format!("void rockfish_hip_add_{tag}(")), "{s}");
        assert!(s.contains("const unsigned short* in0,"), "{s}");
        assert!(s.contains("const unsigned short* in1,"), "{s}");
        assert!(s.contains("    unsigned short* out,"), "{s}");
        let (l, st) = (
            format!("unpopped_{tag}_load"),
            format!("unpopped_{tag}_store"),
        );
        // Both operands widened on load, the sum rounded once on store.
        assert!(
            s.contains(&format!("out[i] = {st}(({l}(in0[i]) + {l}(in1[i])));")),
            "{s}"
        );
        // The codec is device code and uses no macro whose HIP availability is unchecked.
        assert!(
            s.contains(&format!("static __device__ float {l}(unsigned short h)")),
            "{s}"
        );
        assert!(
            s.contains(&format!("static __device__ unsigned short {st}(float x)")),
            "{s}"
        );
        assert!(!s.contains("INFINITY") && !s.contains("NAN"), "{s}");
        assert!(s.contains("__builtin_inff()"), "{s}");
        // No AMD half header or type.
        assert!(!s.contains("__half") && !s.contains("hip_fp16") && !s.contains("__hip_bfloat16"));
    }
}

#[test]
fn only_the_scalar_elementwise_schedule_without_params_is_spelled() {
    for (dt, _) in HALVES {
        // A runtime parameter is a typed refusal (its dtype is not settled).
        let o = OpDef::elementwise("p", 1, &[dt], input(0) * param(0));
        let k = key(1, dt, "rocm:gfx906", 1);
        let plan = try_build_plan_for(&o, &k, &Hip).expect("plan");
        let e = Hip
            .lower(&plan)
            .expect_err("a param at a half must be refused");
        assert!(
            matches!(e, LowerError::UnsupportedPlanShape { .. }),
            "{e:?}"
        );
    }
}

#[test]
fn a_half_binary_op_the_float_path_declines_is_still_declined() {
    // `Max` is spelled at f32; a half goes through the same f32 lowering, so it is spelled too.
    let o = OpDef::elementwise(
        "mx",
        2,
        &[ElementKind::F16],
        input(0).binary(BinaryOp::Max, input(1)),
    );
    let s = try_generate(&o, &key(2, ElementKind::F16, "rocm:gfx906", 1), &Hip)
        .unwrap()
        .source;
    assert!(s.contains("unpopped_f16_load(in0[i])"), "{s}");
}

#[test]
fn an_aligned_half_cell_is_spelled_through_ushort_vectors_lane_by_lane() {
    // 4-byte alignment on a 16-bit dtype plans a 2-wide vector cell.
    for (dt, tag) in HALVES {
        let k = key(2, dt, "rocm:gfx906", 4);
        let o = add(dt);
        let plan = try_build_plan_for(&o, &k, &Hip).expect("plan");
        assert!(
            matches!(plan.schedule, Schedule::Vectorized { width: 2 }),
            "{:?}",
            plan.schedule
        );
        let s = Hip.lower(&plan).expect("lower").source;
        let (l, st) = (
            format!("unpopped_{tag}_load"),
            format!("unpopped_{tag}_store"),
        );
        assert!(
            s.contains("const ushort2* v0 = reinterpret_cast<const ushort2*>(in0);"),
            "{s}"
        );
        for lane in ["x", "y"] {
            assert!(
                s.contains(&format!(
                    "r.{lane} = {st}(({l}(a0.{lane}) + {l}(a1.{lane})));"
                )),
                "{s}"
            );
        }
        // and the scalar tail for the last n % 2 elements
        assert!(
            s.contains(&format!("out[i] = {st}(({l}(in0[i]) + {l}(in1[i])));")),
            "{s}"
        );
    }
}
