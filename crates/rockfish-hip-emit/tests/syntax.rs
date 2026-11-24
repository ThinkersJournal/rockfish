//! Host syntax check of the emitted kernels: each is run through `clang++ -fsyntax-only` against
//! the shim `tests/shim/hip/hip_runtime.h`.
//!
//! **What this shows.** The emitted text parses and type-checks as C++ with the names HIP device
//! code uses (grid-stride loop variables, vector types, math calls). **What it does not show:**
//! anything a HIP compiler adds (address spaces, device builtins, `__half` types), launch
//! behaviour or numerics: the shim is Rockfish-written and permissive. The rig rows in
//! `docs/DESIGN-R7.md` section 6 still stand. A machine without clang skips with a printed
//! message (set `ROCKFISH_CLANG` to name one); with `REQUIRE_CLANG=1` it fails instead.

use std::path::PathBuf;
mod common;

use std::process::Command;

use rockfish_hip_emit::Hip;
use unpopped::backend::Backend;
use unpopped::ir::{BaseOffset, OpDef, ReduceOp, input, konst, reduced};
use unpopped::plan::{KernelPlan, try_build_plan};
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key};

fn gfx906() -> TargetId {
    TargetId::parse("rocm:gfx906").unwrap()
}

fn clang() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ROCKFISH_CLANG") {
        // An empty value means "no clang": it lets a run prove the REQUIRE_CLANG failure on a box that has one.
        return if p.is_empty() {
            None
        } else {
            Some(PathBuf::from(p))
        };
    }
    let candidates = [
        PathBuf::from("clang++"),
        PathBuf::from(
            "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang++.exe",
        ),
    ];
    candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

fn check(label: &str, plan: &KernelPlan<'_>, cc: &PathBuf, n: usize) {
    let src = Hip
        .lower(plan)
        .unwrap_or_else(|e| panic!("{label}: {e:?}"))
        .source;
    let dir = std::env::temp_dir().join("rockfish-syntax");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{n}_{label}.cpp"));
    std::fs::write(&file, &src).unwrap();
    let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/shim");
    let out = Command::new(cc)
        .args([
            "-fsyntax-only",
            "-std=c++17",
            "-x",
            "c++",
            "-Wall",
            "-Wno-unused-variable",
            // A runner's clang can be older than the MSVC STL it finds; this check parses the
            // kernels, it does not link against the STL.
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        ])
        .arg(format!("-I{}", shim.display()))
        .arg(&file)
        .output()
        .expect("clang runs");
    assert!(
        out.status.success(),
        "{label} does not parse:\n{}\n--- source ---\n{src}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn esz(dt: ElementKind) -> u32 {
    match dt {
        ElementKind::F64 | ElementKind::I64 | ElementKind::U64 => 8,
        ElementKind::F16 | ElementKind::Bf16 => 2,
        _ => 4,
    }
}

fn ew_key(
    n_in: usize,
    dt: ElementKind,
    align: u32,
    shape: &[i64],
    strides: &[&[i64]],
) -> StructureKey {
    let mut ops: Vec<OperandDesc> = (0..n_in)
        .map(|i| OperandDesc::new(shape.len(), shape, strides[i], dt, align))
        .collect();
    ops.push(OperandDesc::new(
        shape.len(),
        shape,
        strides[n_in],
        dt,
        align,
    ));
    let cat = if n_in == 1 {
        OpCategory::UnaryElementwise
    } else {
        OpCategory::BinaryElementwise
    };
    structure_key(cat, &ops, gfx906())
}

#[test]
fn every_emitted_cell_parses_as_cpp() {
    let Some(cc) = clang() else {
        common::skip_or_fail(
            "no clang++ found: the host syntax check did not run (set ROCKFISH_CLANG)",
        );
        return;
    };
    let mut n = 0;
    let mut go = |label: &str, o: &OpDef, key: &StructureKey| {
        n += 1;
        let plan = try_build_plan(o, key).unwrap_or_else(|e| panic!("{label}: plan {e:?}"));
        check(label, &plan, &cc, n);
    };
    let float_int = [
        ElementKind::F32,
        ElementKind::F64,
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::U32,
        ElementKind::U64,
        ElementKind::F16,
        ElementKind::Bf16,
    ];
    for dt in float_int {
        // Scalar elementwise (align = element size defeats vectorization).
        let body = if matches!(
            dt,
            ElementKind::I32 | ElementKind::I64 | ElementKind::U32 | ElementKind::U64
        ) {
            input(0) + input(1)
        } else {
            input(0) + input(1) * konst(2.0)
        };
        let o = OpDef::elementwise("ew", 2, &[dt], body);
        let key = ew_key(2, dt, esz(dt), &[1 << 20], &[&[1], &[1], &[1]]);
        go(&format!("scalar_{dt:?}"), &o, &key);

        // Strided: a transposed operand.
        let key = ew_key(2, dt, esz(dt) * 64, &[4, 3], &[&[1, 4], &[3, 1], &[3, 1]]);
        go(&format!("strided_{dt:?}"), &o, &key);

        // Strided with runtime base offsets on input 1 and the output.
        let oo = OpDef::elementwise("ewoff", 2, &[dt], input(0) + input(1)).with_base_offsets(
            vec![BaseOffset::Zero, BaseOffset::Runtime],
            BaseOffset::Runtime,
        );
        let key = ew_key(2, dt, esz(dt) * 64, &[4, 3], &[&[3, 1], &[0, 0], &[3, 1]]);
        go(&format!("offsets_{dt:?}"), &oo, &key);
    }
    // Vectorized float.
    for dt in [ElementKind::F32, ElementKind::F64] {
        let o = OpDef::elementwise("vec", 2, &[dt], input(0) + input(1));
        let key = ew_key(2, dt, 256, &[1 << 20], &[&[1], &[1], &[1]]);
        go(&format!("vec_{dt:?}"), &o, &key);
    }
    // Reduction and scan, float and integer.
    for dt in [ElementKind::F32, ElementKind::I32] {
        let mat = OperandDesc::new(2, &[64, 1024], &[1024, 1], dt, 4);
        let row = OperandDesc::new(1, &[64], &[1], dt, 4);
        for rop in [ReduceOp::Sum, ReduceOp::Max] {
            let o = OpDef::reduction("red", 1, &[dt], input(0), rop);
            let key = structure_key(OpCategory::Reduction, &[mat, row], gfx906());
            go(&format!("reduce_{dt:?}_{rop:?}"), &o, &key);
            let o = OpDef::scan("scn", 1, &[dt], rop, 1, false, true, input(0), reduced(0));
            let key = structure_key(OpCategory::Scan, &[mat, mat], gfx906());
            go(&format!("scan_{dt:?}_{rop:?}"), &o, &key);
        }
    }
    // Sliding-window pool.
    for dt in [ElementKind::F32, ElementKind::I32, ElementKind::F16] {
        let o = OpDef::window(
            "win",
            1,
            &[dt],
            ReduceOp::Max,
            1,
            3,
            2,
            1,
            1,
            1,
            false,
            input(0),
            reduced(0),
        );
        let in_shape = [2i64, 5, 4];
        let out_shape = [2i64, 2, 4];
        let d = [
            OperandDesc::new(3, &in_shape, &[20, 4, 1], dt, esz(dt) * 64),
            OperandDesc::new(3, &out_shape, &[8, 4, 1], dt, esz(dt) * 64),
        ];
        let key = structure_key(OpCategory::Pooling, &d, gfx906());
        go(&format!("window_{dt:?}"), &o, &key);
    }
    eprintln!("{n} kernels parsed with {}", cc.display());
}
