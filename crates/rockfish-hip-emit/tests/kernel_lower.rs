// SPDX-License-Identifier: MIT OR Apache-2.0
//! The S1 kernel-layer lowerer (`src/kernel.rs`): what it writes, what it declines, and a host
//! `clang++ -fsyntax-only` parse of the text.
//!
//! **What this shows:** the exact text for a saxpy-shaped kernel; that a `Store` to a read-only
//! buffer, an unspelled statement, expression, dtype and place are each refused by name; that
//! signed integer arithmetic is emitted wrapping; and that the text of every fixture S1 lowers
//! parses as C++ against the shim header. **What it does not show:** that a HIP compiler accepts
//! it, the launch ABI (a stated contract in `src/kernel.rs`, not checked), or any numerics:
//! UNVERIFIED on a device (no ROCm box). A machine without clang skips (`REQUIRE_CLANG=1`
//! fails instead).

use std::path::PathBuf;
use std::process::Command;

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::fixtures::fixtures;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

mod common;

fn bx() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::BlockIdx(0))
}
fn local(l: u32) -> ScalarExpr {
    ScalarExpr::Local(l)
}
fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

/// `y[i] = a * x[i] + y[i]` for `i = blockIdx.x * blockDim.x + threadIdx.x < n`.
fn saxpy() -> KernelFn {
    let i = ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(bx()),
            Box::new(ScalarExpr::Exec(ExecId::BlockDim(0))),
        )),
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
    );
    let at_i = |buffer| Place::Logical {
        buffer,
        coords: vec![local(3)],
    };
    KernelFn::new(
        "saxpy",
        ElementKind::F32,
        vec![
            Param::buffer("x", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("y", ElementKind::F32, BufferRole::ReadWrite, Some(1)),
            Param::scalar("a", ElementKind::F32, 0),
            Param::scalar("n", ElementKind::U32, 1),
        ],
        vec![
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::F32),
        ],
        Block(vec![
            Stmt::Let { local: 3, init: i },
            Stmt::If {
                cond: bin(BinaryOp::CmpLt, local(3), local(1)),
                then: Block(vec![
                    Stmt::Load {
                        local: 4,
                        place: at_i(0),
                    },
                    Stmt::Load {
                        local: 5,
                        place: at_i(1),
                    },
                    Stmt::Store {
                        place: at_i(1),
                        value: ScalarExpr::Add(
                            Box::new(ScalarExpr::Mul(Box::new(local(0)), Box::new(local(4)))),
                            Box::new(local(5)),
                        ),
                    },
                ]),
                els: Block(vec![]),
            },
        ]),
    )
    .with_launch(LaunchShape::new([256, 1, 1], [64, 1, 1]))
}

fn detail(k: &KernelFn) -> String {
    match Hip.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected UnsupportedKernel, got {other:?}"),
    }
}

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
        PathBuf::from(
            "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang++.exe",
        ),
    ]
    .into_iter()
    .find(|c| Command::new(c).arg("--version").output().is_ok_and(|o| o.status.success()))
}

fn parses(label: &str, src: &str) {
    let Some(cc) = clang() else {
        common::skip_or_fail("no clang++ found");
        return;
    };
    let dir = std::env::temp_dir().join("rockfish-kernel-lower");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{label}.cpp"));
    std::fs::write(&file, src).unwrap();
    let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/shim");
    let out = Command::new(cc)
        .args([
            "-fsyntax-only",
            "-std=c++17",
            "-x",
            "c++",
            "-Wall",
            "-Wno-unused-variable",
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

#[test]
fn saxpy_text_and_abi() {
    let g = Hip.lower_kernel(&saxpy()).expect("saxpy lowers");
    let s = &g.source;
    assert_eq!(g.name, "rockfish_hip_k_saxpy");
    assert!(
        s.contains("extern \"C\" __global__ void rockfish_hip_k_saxpy("),
        "{s}"
    );
    assert!(
        s.contains("const float* __restrict__ p0,\n    long long p0_s0,"),
        "{s}"
    );
    assert!(
        s.contains("float* __restrict__ p1,\n    long long p1_s0,"),
        "{s}"
    );
    assert!(s.contains("float l0,\n    unsigned int l1"), "{s}");
    assert!(
        s.contains("if ((((unsigned int)((l3) < (l1) ? 1 : 0))) != 0) {"),
        "{s}"
    );
    assert!(
        s.contains("p1[(long long)(l3) * p1_s0] = (l0 * l4) + l5;")
            || s.contains("= ((l0 * l4) + l5);"),
        "{s}"
    );
    parses("saxpy", s);
}

#[test]
fn a_store_to_a_read_only_buffer_is_refused() {
    let mut k = saxpy();
    // Mutant: point the store at buffer 0 (`Read`). Must be refused, not emitted.
    if let Stmt::If { then, .. } = &mut k.body.0[1]
        && let Stmt::Store { place, .. } = &mut then.0[2]
    {
        *place = Place::Logical {
            buffer: 0,
            coords: vec![local(3)],
        };
    }
    let d = detail(&k);
    // The checker is the first gate and names it; the lowerer's own role check is a second.
    assert!(d.contains("WriteToReadOnly"), "{d}");
}

#[test]
fn unspelled_things_are_refused_by_name() {
    let mut k = saxpy();
    k.body.0.push(Stmt::MaskedStore {
        place: Place::Logical {
            buffer: 1,
            coords: vec![local(3)],
        },
        value: ScalarExpr::Const(0.0),
        mask: ScalarExpr::Const(1.0),
    });
    assert!(detail(&k).contains("statement `MaskedStore`"));

    let mut k = saxpy();
    k.body.0.push(Stmt::Let {
        local: 2,
        init: ScalarExpr::Unary(unpopped::ir::UnaryOp::Neg, Box::new(ScalarExpr::Const(1.0))),
    });
    let d = detail(&k);
    assert!(d.contains("expression `Unary`"));

    let mut k = saxpy();
    // An unused trailing F16 buffer: checker-valid, so the refusal is the lowerer's.
    k.params.push(Param::buffer(
        "h",
        ElementKind::F16,
        BufferRole::Read,
        Some(1),
    ));
    assert!(detail(&k).contains("dtype F16"));

    let mut k = saxpy();
    k.body.0.push(Stmt::Store {
        place: Place::Raw {
            buffer: 1,
            byte_offset: ScalarExpr::ConstInt(0, ElementKind::I32),
            elem: ElementKind::F32,
        },
        value: ScalarExpr::Const(0.0),
    });
    // `Raw` is caught earlier, by the shared unlowered-place scan, with its own reason.
    assert!(detail(&k).contains("Raw"));
}

#[test]
fn signed_integer_arithmetic_wraps_through_the_unsigned_twin() {
    let k = KernelFn::new(
        "wrap",
        ElementKind::I32,
        vec![],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(vec![Stmt::Let {
            local: 0,
            init: ScalarExpr::Add(
                Box::new(ScalarExpr::ConstInt(i64::from(i32::MAX), ElementKind::I32)),
                Box::new(ScalarExpr::ConstInt(1, ElementKind::I32)),
            ),
        }]),
    );
    let s = Hip.lower_kernel(&k).expect("lowers").source;
    assert!(
        s.contains("((int)((unsigned int)(((int)2147483647LL)) + (unsigned int)(((int)1LL))))"),
        "{s}"
    );
    parses("wrap", &s);
}

#[test]
fn every_fixture_s1_lowers_parses() {
    let mut n = 0;
    for (name, fx) in fixtures() {
        for f in fx.functions() {
            if let Ok(g) = Hip.lower_kernel(f) {
                parses(&format!("fixture_{name}"), &g.source);
                n += 1;
            }
        }
    }
    assert_eq!(n, 10, "the S1-lowered fixture set changed: a decision");
}
