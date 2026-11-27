// SPDX-License-Identifier: MIT OR Apache-2.0
//! Value-level host run of the S1c `Cast` lowering: float to integer (`TowardZero`, `Clamp`, NaN
//! to 0) and integer to float.
//!
//! Each kernel is emitted by `Hip::lower_kernel`, compiled with host `clang++` against the shim
//! (the grid variables are plain globals, so `main` plays each thread by setting `threadIdx.x`),
//! run on boundary vectors, and compared with Rust's own `as` casts, which saturate, truncate
//! toward zero and map NaN to 0 (the ruled rule, independent of the emitted text).
//!
//! **What this shows:** the emitted expression gives the ruled results under a host C++ compiler.
//! **What it does not:** hipcc, the device, or GPU codegen of the same text (rig rows stand).
//! Skips with a printed message when no `clang++` is found; `REQUIRE_CLANG=1` fails instead.

use std::path::PathBuf;
use std::process::Command;

use rockfish_hip_emit::Hip;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

mod common;

fn clang() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ROCKFISH_CLANG") {
        return if p.is_empty() {
            None
        } else {
            Some(PathBuf::from(p))
        };
    }
    [
        PathBuf::from("clang++"),
        PathBuf::from("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang++.exe"),
    ]
    .into_iter()
    .find(|c| Command::new(c).arg("--version").output().is_ok_and(|o| o.status.success()))
}

fn cast_kernel(from: ElementKind, to: ElementKind, round: RoundMode, sat: Saturation) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Local(0)],
    };
    KernelFn::new(
        "cast",
        from,
        vec![
            Param::buffer("x", from, BufferRole::Read, Some(1)),
            Param::buffer("y", to, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(from),
            LocalDecl::scalar(to),
        ],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
            },
            Stmt::Load {
                local: 1,
                place: at(0),
            },
            Stmt::Let {
                local: 2,
                init: ScalarExpr::Cast(Box::new(ScalarExpr::Local(1)), CastSpec { to, round, sat }),
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Local(2),
            },
        ]),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
}

const INPUTS: [f64; 24] = [
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.9,
    -1.9,
    2147483520.0,
    2147483647.0,
    2147483648.0,
    -2147483648.0,
    -2147483904.0,
    4294967040.0,
    4294967295.0,
    4294967296.0,
    9.2e18,
    9.223372036854776e18,
    -9.3e18,
    1.8e19,
    1.8446744073709552e19,
    3.0e9,
    -1.0,
];

/// Run one kernel; returns each output printed as a decimal integer.
fn run(from: ElementKind, to: ElementKind, cc: &PathBuf, tag: &str) -> Vec<String> {
    let k = cast_kernel(from, to, RoundMode::TowardZero, Saturation::Clamp);
    let src = Hip.lower_kernel(&k).expect("lowers").source;
    let (ft, tt) = (
        if from == ElementKind::F32 {
            "float"
        } else {
            "double"
        },
        match to {
            ElementKind::I32 => "int",
            ElementKind::U32 => "unsigned int",
            ElementKind::I64 => "long long",
            _ => "unsigned long long",
        },
    );
    let lits: Vec<String> = INPUTS
        .iter()
        .map(|v| {
            if v.is_nan() {
                format!("({ft})__builtin_nan(\"\")")
            } else if v.is_infinite() {
                format!("({ft}){}__builtin_inf()", if *v < 0.0 { "-" } else { "" })
            } else {
                format!("({ft}){v:?}")
            }
        })
        .collect();
    let main = format!(
        "{src}\n#include <cstdio>\nint main() {{\n  {ft} in[{n}] = {{{lits}}};\n  {tt} out[{n}];\n  for (unsigned i = 0; i < {n}; ++i) {{ threadIdx.x = i; rockfish_hip_k_cast(in, 1, out, 1); }}\n  for (unsigned i = 0; i < {n}; ++i) std::printf(\"%s\\n\", std::to_string(out[i]).c_str());\n}}\n",
        n = INPUTS.len(),
        lits = lits.join(", ")
    );
    let main = main.replace("#include <cstdio>", "#include <cstdio>\n#include <string>");
    let dir = std::env::temp_dir().join("rockfish-cast-run");
    std::fs::create_dir_all(&dir).unwrap();
    let cpp = dir.join(format!("{tag}.cpp"));
    let exe = dir.join(format!("{tag}.exe"));
    std::fs::write(&cpp, main).unwrap();
    let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/shim");
    let out = Command::new(cc)
        .args([
            "-std=c++17",
            "-x",
            "c++",
            "-O0",
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        ])
        .arg(format!("-I{}", shim.display()))
        .arg(&cpp)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("clang runs");
    assert!(
        out.status.success(),
        "{tag}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().expect("exe runs");
    assert!(run.status.success(), "{tag}: crashed");
    String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

fn expect<T: ToString>(from: ElementKind, f: impl Fn(f64) -> T) -> Vec<String> {
    INPUTS
        .iter()
        .map(|&v| {
            // The kernel's input is the value rounded to the source float type.
            let v = if from == ElementKind::F32 {
                f64::from(v as f32)
            } else {
                v
            };
            f(v).to_string()
        })
        .collect()
}

#[test]
fn float_to_int_cast_matches_rusts_saturating_truncating_as() {
    let Some(cc) = clang() else {
        common::skip_or_fail("no clang++ found");
        return;
    };
    for from in [ElementKind::F32, ElementKind::F64] {
        // Rust's `as` from the (already f32-rounded) value: the ruled rule, independent of the text.
        let cases: [(ElementKind, Vec<String>); 4] = [
            (
                ElementKind::I32,
                expect(from, |v| {
                    if from == ElementKind::F32 {
                        (v as f32) as i32
                    } else {
                        v as i32
                    }
                }),
            ),
            (
                ElementKind::U32,
                expect(from, |v| {
                    if from == ElementKind::F32 {
                        (v as f32) as u32
                    } else {
                        v as u32
                    }
                }),
            ),
            (
                ElementKind::I64,
                expect(from, |v| {
                    if from == ElementKind::F32 {
                        (v as f32) as i64
                    } else {
                        v as i64
                    }
                }),
            ),
            (
                ElementKind::U64,
                expect(from, |v| {
                    if from == ElementKind::F32 {
                        (v as f32) as u64
                    } else {
                        v as u64
                    }
                }),
            ),
        ];
        for (to, want) in cases {
            let got = run(from, to, &cc, &format!("{from:?}_{to:?}"));
            assert_eq!(got, want, "{from:?} -> {to:?}: inputs {INPUTS:?}");
        }
    }
}

#[test]
fn other_casts_and_modes_are_refused_by_name() {
    let refused = |k: KernelFn| {
        matches!(
            Hip.lower_kernel(&k),
            Err(unpopped::backend::LowerError::UnsupportedKernel { .. })
        )
    };
    // float to int needs TowardZero with Clamp; anything else is refused.
    assert!(refused(cast_kernel(
        ElementKind::F32,
        ElementKind::I32,
        RoundMode::NearestEven,
        Saturation::Clamp
    )));
    assert!(refused(cast_kernel(
        ElementKind::F32,
        ElementKind::I32,
        RoundMode::TowardZero,
        Saturation::Wrap
    )));
    // int to float: C's conversion, NearestEven only (text only here; the conversion is the language's).
    let i2f = cast_kernel(
        ElementKind::I32,
        ElementKind::F32,
        RoundMode::NearestEven,
        Saturation::Wrap,
    );
    let t = Hip.lower_kernel(&i2f).expect("int to float lowers").source;
    assert!(t.contains("float l2 = ((float)(l1));"), "{t}");
    assert!(refused(cast_kernel(
        ElementKind::I32,
        ElementKind::F32,
        RoundMode::TowardZero,
        Saturation::Wrap
    )));
    // float to float and int to int are not spelled
    assert!(refused(cast_kernel(
        ElementKind::F32,
        ElementKind::F64,
        RoundMode::NearestEven,
        Saturation::Wrap
    )));
    // control: the spelled combination lowers
    assert!(!refused(cast_kernel(
        ElementKind::F32,
        ElementKind::I32,
        RoundMode::TowardZero,
        Saturation::Clamp
    )));
}
