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
    // `MaskedStore` was the control until S1b spelled it; `GridDepWait` is unspelled still.
    k.body.0.push(Stmt::GridDepWait);
    assert!(detail(&k).contains("statement `GridDepWait`"));

    let mut k = saxpy();
    k.body.0.push(Stmt::Let {
        local: 2,
        init: ScalarExpr::Unary(
            unpopped::ir::UnaryOp::Sqrt,
            Box::new(ScalarExpr::Const(1.0)),
        ),
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
    assert_eq!(
        n, 13,
        "the S1-lowered fixture set changed: a decision (S1c added Barrier)"
    );
}

/// A one-buffer-in, one-buffer-out F32 kernel whose body is `body` over locals `0..=2` (F32) and
/// `3` (U32 thread index, `Let` first).
fn simple(body: Vec<Stmt>) -> KernelFn {
    let mut stmts = vec![Stmt::Let {
        local: 3,
        init: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
    }];
    stmts.extend(body);
    KernelFn::new(
        "s1b",
        ElementKind::F32,
        vec![
            Param::buffer("x", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("y", ElementKind::F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::U32),
        ],
        Block(stmts),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![local(3)],
    }
}

#[test]
fn masked_load_and_store_touch_no_memory_when_the_mask_is_false() {
    let k = simple(vec![
        Stmt::MaskedLoad {
            local: 0,
            place: at(0),
            mask: bin(
                BinaryOp::CmpLt,
                local(3),
                ScalarExpr::ConstInt(10, ElementKind::U32),
            ),
            fill: ScalarExpr::Const(-1.0),
        },
        Stmt::MaskedStore {
            place: at(1),
            value: local(0),
            mask: bin(
                BinaryOp::CmpLt,
                local(3),
                ScalarExpr::ConstInt(10, ElementKind::U32),
            ),
        },
    ]);
    let s = Hip.lower_kernel(&k).expect("lowers").source;
    // The load is a ternary: the memory arm is not evaluated when the mask is false.
    assert!(s.contains("float l0 = ((((unsigned int)((l3) < (((unsigned int)10LL)) ? 1 : 0))) != 0) ? p0[(long long)(l3) * p0_s0] : (-1.0f);"), "{s}");
    assert!(s.contains("{ p1[(long long)(l3) * p1_s0] = l0; }"), "{s}");
    parses("masked", &s);
    // Mutant control: a masked store to the read-only buffer is refused.
    let mut bad = k.clone();
    bad.body.0[2] = Stmt::MaskedStore {
        place: at(0),
        value: local(0),
        mask: ScalarExpr::Const(1.0),
    };
    assert!(detail(&bad).contains("WriteToReadOnly"), "{}", detail(&bad));
}

#[test]
fn float_max_min_are_nan_propagating_neg_abs_select_are_spelled() {
    let k = simple(vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Let {
            local: 1,
            init: bin(BinaryOp::Max, local(0), ScalarExpr::Const(0.0)),
        },
        Stmt::Let {
            local: 2,
            init: bin(BinaryOp::Min, local(0), local(1)),
        },
        Stmt::Assign {
            local: 2,
            value: ScalarExpr::Unary(unpopped::ir::UnaryOp::Neg, Box::new(local(2))),
        },
        Stmt::Assign {
            local: 1,
            value: ScalarExpr::Select(
                Box::new(local(0)),
                Box::new(ScalarExpr::Unary(
                    unpopped::ir::UnaryOp::Abs,
                    Box::new(local(2)),
                )),
                Box::new(ScalarExpr::Const(0.0)),
            ),
        },
        Stmt::Store {
            place: at(1),
            value: local(1),
        },
    ]);
    let s = Hip.lower_kernel(&k).expect("lowers").source;
    // NaN-propagating, the first operand on ties: `>=` for Max, `<=` for Min.
    assert!(
        s.contains("l0 != l0 ? l0 : (0.0f != 0.0f ? 0.0f : (l0 >= 0.0f ? l0 : 0.0f))"),
        "{s}"
    );
    assert!(
        s.contains("l0 != l0 ? l0 : (l1 != l1 ? l1 : (l0 <= l1 ? l0 : l1))"),
        "{s}"
    );
    assert!(s.contains("(-l2)") && s.contains("fabsf(l2)"), "{s}");
    assert!(s.contains("(l0) != 0 ? (fabsf(l2)) : (0.0f)"), "{s}");
    parses("float_ops", &s);
}

#[test]
fn integer_neg_wraps_and_integer_abs_and_select_are_refused() {
    let neg = |ek| {
        KernelFn::new(
            "neg",
            ek,
            vec![],
            vec![LocalDecl::scalar(ek)],
            Block(vec![Stmt::Let {
                local: 0,
                init: ScalarExpr::Unary(
                    unpopped::ir::UnaryOp::Neg,
                    Box::new(ScalarExpr::ConstInt(5, ek)),
                ),
            }]),
        )
    };
    let s = Hip
        .lower_kernel(&neg(ElementKind::I32))
        .expect("i32")
        .source;
    assert!(s.contains("(int)(0u - (unsigned int)(((int)5LL)))"), "{s}");
    parses("neg_i32", &s);
    assert!(Hip.lower_kernel(&neg(ElementKind::U64)).is_ok());

    let abs = KernelFn::new(
        "abs",
        ElementKind::I32,
        vec![],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(vec![Stmt::Let {
            local: 0,
            init: ScalarExpr::Unary(
                unpopped::ir::UnaryOp::Abs,
                Box::new(ScalarExpr::ConstInt(5, ElementKind::I32)),
            ),
        }]),
    );
    assert!(detail(&abs).contains("integer `Abs`"), "{}", detail(&abs));
}

use unpopped::exec::{Extent, MemorySpace};
use unpopped::kernel::{BarrierKind, BufferRole as Role};

/// `tile[tid] = x[tid]; barrier; y[tid] = tile[63 - tid]` over one block of 64 threads: a
/// static `Shared` array and a plain block barrier.
fn reverse_in_shared(barrier: Stmt, alloc: unpopped::exec::AllocDecl) -> KernelFn {
    let tile = |index| Place::Alloc { alloc: 0, index };
    KernelFn::new(
        "rev",
        ElementKind::F32,
        vec![
            Param::buffer("x", ElementKind::F32, Role::Read, Some(1)),
            Param::buffer("y", ElementKind::F32, Role::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(ElementKind::F32),
            LocalDecl::scalar(ElementKind::F32),
        ],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
            },
            Stmt::Load {
                local: 1,
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![local(0)],
                },
            },
            Stmt::Store {
                place: tile(local(0)),
                value: local(1),
            },
            barrier,
            Stmt::Load {
                local: 2,
                place: tile(ScalarExpr::Sub(
                    Box::new(ScalarExpr::ConstInt(63, ElementKind::U32)),
                    Box::new(local(0)),
                )),
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 1,
                    coords: vec![local(0)],
                },
                value: local(1),
            },
        ]),
    )
    .with_allocs(vec![alloc])
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
}

fn shared64() -> unpopped::exec::AllocDecl {
    unpopped::exec::AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Static(64),
    )
}
fn block_barrier() -> Stmt {
    Stmt::Barrier {
        scope: unpopped::exec::Scope::Block,
        kind: BarrierKind::Full,
    }
}

#[test]
fn a_static_shared_array_and_a_block_barrier_lower() {
    let s = Hip
        .lower_kernel(&reverse_in_shared(block_barrier(), shared64()))
        .expect("lowers")
        .source;
    assert!(s.contains("    __shared__ float s0[64];"), "{s}");
    assert!(s.contains("s0[(long long)(l0)] = l1;"), "{s}");
    assert!(s.contains("    __syncthreads();"), "{s}");
    // The shared array is declared before the body; the barrier sits between store and load.
    assert!(
        s.find("__shared__").unwrap() < s.find("__syncthreads").unwrap(),
        "{s}"
    );
    assert!(
        s.find("s0[(long long)(l0)] = l1;").unwrap() < s.find("__syncthreads").unwrap(),
        "{s}"
    );
    parses("shared_rev", &s);
}

#[test]
fn other_barriers_and_other_allocations_are_refused_by_name() {
    let named = Stmt::Barrier {
        scope: unpopped::exec::Scope::Block,
        kind: BarrierKind::Named { id: 1, count: 64 },
    };
    let d = detail(&reverse_in_shared(named, shared64()));
    assert!(
        d.contains("Barrier") && d.contains("only Block/Full"),
        "{d}"
    );
    let sub = Stmt::Barrier {
        scope: unpopped::exec::Scope::Subgroup,
        kind: BarrierKind::Full,
    };
    assert!(detail(&reverse_in_shared(sub, shared64())).contains("Subgroup"));
    let private = unpopped::exec::AllocDecl::new(
        "t",
        MemorySpace::Private,
        ElementKind::F32,
        Extent::Static(64),
    );
    let d = detail(&reverse_in_shared(block_barrier(), private));
    assert!(d.contains("only static Shared"), "{d}");
    let dynamic = unpopped::exec::AllocDecl::new(
        "t",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Param(0),
    );
    // A dynamic extent needs a scalar parameter the checker can see; whichever gate fires, it is a refusal.
    assert!(matches!(
        Hip.lower_kernel(&reverse_in_shared(block_barrier(), dynamic)),
        Err(LowerError::UnsupportedKernel { .. })
    ));
    // Control: the same kernel with the plain barrier and a static Shared array lowers.
    assert!(
        Hip.lower_kernel(&reverse_in_shared(block_barrier(), shared64()))
            .is_ok()
    );
}
