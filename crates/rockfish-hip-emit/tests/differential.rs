// SPDX-License-Identifier: MIT OR Apache-2.0
//! Host differential: the lowered HIP text, run on the host, against Unpopped's oracle.
//!
//! Each case is a `KernelFn`. The oracle runs it (`run_launch_with_subgroup`). `Hip` lowers it,
//! and the text is compiled with host `clang++` against `tests/shim_threads/hip/hip_runtime.h`
//! (one host thread per HIP thread, `__syncthreads` as a block barrier, `__shfl*`/`__all`/`__any`/
//! `__ballot` as slot exchanges per subgroup) and run on the same inputs. Every buffer is then
//! compared byte for byte (floats: equal bits, or both NaN).
//!
//! **What this shows:** the *lowering* (Max/Min NaN and tie rules, `Cast`, masked operations,
//! the shuffle edge rules Idx mod n and Up/Down/Xor own-value, votes, integer reductions,
//! broadcasts, shared memory with a barrier) means the same as the oracle, at subgroup sizes 32
//! and 64. **What it does not show:** the hardware. The subgroup operations in the shim are
//! OUR MODEL of AMD's, written from the HIP docs; no device has confirmed them: UNMEASURED.
//! Vacuity: for each class a hand mutant edits the emitted text (anchor counted: exactly one
//! hit) and the differential must then FAIL.
//!
//! Skips without `clang++`; `REQUIRE_CLANG=1` fails instead.

use std::path::PathBuf;
use std::process::Command;

use rockfish_hip_emit::Hip;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, ReduceOp, RoundMode, Saturation, ScalarExpr, UnaryOp};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::{ElementKind, TargetId};

mod common;

const E: ElementKind = ElementKind::F32;

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

fn ctype(ek: ElementKind) -> &'static str {
    match ek {
        ElementKind::F32 => "float",
        ElementKind::F64 => "double",
        ElementKind::I32 => "int",
        ElementKind::U32 => "unsigned int",
        ElementKind::I64 => "long long",
        ElementKind::U64 => "unsigned long long",
        other => panic!("{other:?}"),
    }
}
fn width(ek: ElementKind) -> usize {
    match ek {
        ElementKind::F64 | ElementKind::I64 | ElementKind::U64 => 8,
        _ => 4,
    }
}

/// Dense one-axis buffer from raw little-endian bytes.
fn buf(dt: ElementKind, bytes: Vec<u8>) -> KernelArg {
    let n = (bytes.len() / width(dt)) as i64;
    KernelArg::Buffer(TypedBuffer::new(dt, vec![n], vec![1], bytes))
}
fn f32s(v: &[f32]) -> KernelArg {
    buf(
        ElementKind::F32,
        v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
    )
}
fn f64s(v: &[f64]) -> KernelArg {
    buf(
        ElementKind::F64,
        v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
    )
}
fn i32s(v: &[i32]) -> KernelArg {
    buf(
        ElementKind::I32,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}
fn zeros(dt: ElementKind, n: usize) -> KernelArg {
    buf(dt, vec![0; n * width(dt)])
}

/// One differential case. `mutate` edits the emitted text: `(anchor, replacement)`, the anchor
/// must occur exactly once.
struct Case<'a> {
    k: &'a KernelFn,
    args: Vec<KernelArg>,
    wave: u32,
    mutate: &'a [(&'a str, &'a str)],
}

fn bytes_of(a: &KernelArg) -> &TypedBuffer {
    match a {
        KernelArg::Buffer(b) => b,
        other => panic!("{other:?}"),
    }
}

fn same(dt: ElementKind, a: &[u8], b: &[u8]) -> bool {
    if a == b {
        return true;
    }
    let w = width(dt);
    if !matches!(dt, ElementKind::F32 | ElementKind::F64) || a.len() != b.len() {
        return false;
    }
    a.chunks(w).zip(b.chunks(w)).all(|(x, y)| {
        x == y
            || if w == 4 {
                f32::from_le_bytes(x.try_into().unwrap()).is_nan()
                    && f32::from_le_bytes(y.try_into().unwrap()).is_nan()
            } else {
                f64::from_le_bytes(x.try_into().unwrap()).is_nan()
                    && f64::from_le_bytes(y.try_into().unwrap()).is_nan()
            }
    })
}

/// `Ok(())` when the host run matches the oracle on every buffer; `Err(report)` otherwise.
fn differential(label: &str, c: &Case) -> Result<(), String> {
    let Some(cc) = clang() else {
        common::skip_or_fail("no clang++ found");
        return Ok(());
    };
    let launch = c.k.launch.expect("a launched kernel");
    // Oracle.
    let mut oracle = c.args.clone();
    run_launch_with_subgroup(c.k, &mut oracle, Some(c.wave))
        .map_err(|e| format!("{label}: oracle trap {e:?}"))?;
    // Lower.
    let arch = if c.wave == 64 { "gfx90a" } else { "gfx1100" };
    let g = Hip
        .lower_kernel_for(c.k, &TargetId::parse(&format!("rocm:{arch}")).unwrap())
        .map_err(|e| format!("{label}: lowering {e:?}"))?;
    let mut src = g.source.clone();
    for (anchor, rep) in c.mutate {
        assert_eq!(
            src.matches(anchor).count(),
            1,
            "{label}: mutation anchor `{anchor}` must hit once:\n{src}"
        );
        src = src.replace(anchor, rep);
    }
    // Host program.
    let mut decls = String::new();
    let mut call = Vec::new();
    for (i, a) in c.args.iter().enumerate() {
        match a {
            KernelArg::Buffer(b) => {
                let hex: Vec<String> = b.bytes.iter().map(|x| format!("0x{x:02x}")).collect();
                let n = b.bytes.len() / width(b.dtype);
                decls += &format!(
                    "  static const unsigned char init{i}[] = {{{}}};\n  {t} buf{i}[{n}]; std::memcpy(buf{i}, init{i}, sizeof init{i});\n",
                    hex.join(","),
                    t = ctype(b.dtype)
                );
                call.push(format!("buf{i}"));
                if src.contains(&format!("p{i}_s0")) {
                    call.push("1".to_string());
                }
            }
            KernelArg::Float(v) => call.push(format!("{v:?}")),
            KernelArg::Int(v) => call.push(format!("{v}")),
            other => panic!("{other:?}"),
        }
    }
    let mut dump = String::new();
    for (i, a) in c.args.iter().enumerate() {
        if let KernelArg::Buffer(_) = a {
            dump += &format!(
                "  for (unsigned i = 0; i < sizeof buf{i}; ++i) std::printf(\"%02x\", ((unsigned char*)buf{i})[i]);\n  std::printf(\"\\n\");\n"
            );
        }
    }
    let main = format!(
        "{src}\n#include <cstdio>\nint main() {{\n{decls}  shim_launch({bx},{by},{bz},{gx},{gy},{gz}, [&] {{ rockfish_hip_k_{name}({args}); }});\n{dump}  return 0;\n}}\n",
        bx = launch.block[0],
        by = launch.block[1],
        bz = launch.block[2],
        gx = launch.grid[0],
        gy = launch.grid[1],
        gz = launch.grid[2],
        name = c.k.name,
        args = call.join(", "),
    );
    let dir = std::env::temp_dir().join("rockfish-differential");
    std::fs::create_dir_all(&dir).unwrap();
    let cpp = dir.join(format!("{label}.cpp"));
    let exe = dir.join(format!("{label}.exe"));
    std::fs::write(&cpp, &main).unwrap();
    let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/shim_threads");
    let out = Command::new(cc)
        .args([
            "-std=c++17",
            "-x",
            "c++",
            "-O0",
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        ])
        .arg(format!("-DSHIM_WAVE={}", c.wave))
        .arg(format!("-I{}", shim.display()))
        .arg(&cpp)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("clang runs");
    if !out.status.success() {
        return Err(format!(
            "{label}: host compile failed:\n{}\n{main}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let run = Command::new(&exe).output().expect("exe runs");
    if !run.status.success() {
        return Err(format!("{label}: host run crashed"));
    }
    let text = String::from_utf8_lossy(&run.stdout).to_string();
    let lines: Vec<&str> = text.lines().collect();
    let mut li = 0;
    for (i, a) in c.args.iter().enumerate() {
        if let KernelArg::Buffer(_) = a {
            let want = bytes_of(&oracle[i]);
            let got: Vec<u8> = (0..lines[li].len() / 2)
                .map(|j| u8::from_str_radix(&lines[li][2 * j..2 * j + 2], 16).unwrap())
                .collect();
            li += 1;
            if !same(want.dtype, &want.bytes, &got) {
                let w = width(want.dtype);
                let first = want
                    .bytes
                    .chunks(w)
                    .zip(got.chunks(w))
                    .position(|(x, y)| x != y)
                    .unwrap_or(0);
                return Err(format!(
                    "{label}: buffer {i} differs, first at element {first}: oracle {:02x?} host {:02x?}",
                    &want.bytes[first * w..first * w + w],
                    &got[first * w..first * w + w]
                ));
            }
        }
    }
    Ok(())
}

/// The case must match; and with each mutation the same case must NOT match.
fn check(label: &str, k: &KernelFn, args: &[KernelArg], wave: u32, mutants: &[&[(&str, &str)]]) {
    let case = Case {
        k,
        args: args.to_vec(),
        wave,
        mutate: &[],
    };
    if let Err(e) = differential(label, &case) {
        panic!("{e}");
    }
    for (n, m) in mutants.iter().enumerate() {
        let mc = Case {
            k,
            args: args.to_vec(),
            wave,
            mutate: m,
        };
        // The mutant must be caught by a VALUE difference, not by failing to compile or run.
        match differential(&format!("{label}_mut{n}"), &mc) {
            Err(e) if e.contains("differs") => {}
            other => {
                panic!("{label}: mutant {n} {m:?} was not caught as a value difference: {other:?}")
            }
        }
    }
}

// ---------- kernel builders ----------

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}
fn loc(l: u32) -> ScalarExpr {
    ScalarExpr::Local(l)
}
fn at(buffer: u32, index: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![index],
    }
}
fn plus(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn ku32(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::U32)
}
fn ki32(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::I32)
}
fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}
fn lanes(n: u32) -> LaunchShape {
    LaunchShape::new([n * 2, 1, 1], [1, 1, 1]).with_subgroup(n, n)
}

// ---------- classes ----------

#[test]
fn max_min_nan_and_ties() {
    // out0 = Max(x, y), out1 = Min(x, y) over NaN, +-0 ties, infinities.
    let x = [
        f32::NAN,
        1.0,
        -0.0,
        0.0,
        f32::INFINITY,
        2.0,
        f32::NEG_INFINITY,
        0.0,
    ];
    let y = [
        1.0,
        f32::NAN,
        0.0,
        -0.0,
        f32::NEG_INFINITY,
        2.0,
        f32::NEG_INFINITY,
        0.0,
    ];
    let k = KernelFn::new(
        "mm",
        E,
        vec![
            Param::buffer("x", E, BufferRole::Read, Some(1)),
            Param::buffer("y", E, BufferRole::Read, Some(1)),
            Param::buffer("mx", E, BufferRole::Write, Some(1)),
            Param::buffer("mn", E, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(E),
            LocalDecl::scalar(E),
            LocalDecl::scalar(E),
            LocalDecl::scalar(E),
        ],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: tid(),
            },
            Stmt::Load {
                local: 1,
                place: at(0, loc(0)),
            },
            Stmt::Load {
                local: 2,
                place: at(1, loc(0)),
            },
            Stmt::Let {
                local: 3,
                init: bin(BinaryOp::Max, loc(1), loc(2)),
            },
            Stmt::Let {
                local: 4,
                init: bin(BinaryOp::Min, loc(1), loc(2)),
            },
            Stmt::Store {
                place: at(2, loc(0)),
                value: loc(3),
            },
            Stmt::Store {
                place: at(3, loc(0)),
                value: loc(4),
            },
        ]),
    )
    .with_launch(lanes(32));
    // A block of 64 threads (two subgroups of 32): the inputs are padded with a benign pair.
    let pad = |v: &[f32]| {
        let mut p = v.to_vec();
        p.resize(64, 1.5);
        p
    };
    let args = [f32s(&pad(&x)), f32s(&pad(&y)), zeros(E, 64), zeros(E, 64)];
    check("maxmin", &k, &args, 32, &[&[("l1 >= l2", "l1 > l2")]]);
}

#[test]
fn casts_float_to_int_and_int_to_float() {
    let xs: Vec<f64> = vec![
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
    let mut xs = xs;
    xs.resize(64, 7.25);
    let ints: Vec<i32> = (0..64)
        .map(|i| {
            if i % 2 == 0 {
                i * 33_554_433
            } else {
                -i * 16_777_217
            }
        })
        .collect();
    let spec = |to| CastSpec {
        to,
        round: RoundMode::TowardZero,
        sat: Saturation::Clamp,
    };
    let tys = [
        ElementKind::I32,
        ElementKind::U32,
        ElementKind::I64,
        ElementKind::U64,
    ];
    let mut params = vec![
        Param::buffer("x", ElementKind::F64, BufferRole::Read, Some(1)),
        Param::buffer("n", ElementKind::I32, BufferRole::Read, Some(1)),
    ];
    let mut locals = vec![
        LocalDecl::scalar(ElementKind::U32),
        LocalDecl::scalar(ElementKind::F64),
        LocalDecl::scalar(ElementKind::I32),
    ];
    let mut body = vec![
        Stmt::Let {
            local: 0,
            init: tid(),
        },
        Stmt::Load {
            local: 1,
            place: at(0, loc(0)),
        },
        Stmt::Load {
            local: 2,
            place: at(1, loc(0)),
        },
    ];
    for (j, &t) in tys.iter().enumerate() {
        params.push(Param::buffer(
            format!("o{j}"),
            t,
            BufferRole::Write,
            Some(1),
        ));
        locals.push(LocalDecl::scalar(t));
        let l = 3 + j as u32;
        body.push(Stmt::Let {
            local: l,
            init: ScalarExpr::Cast(Box::new(loc(1)), spec(t)),
        });
        body.push(Stmt::Store {
            place: at(2 + j as u32, loc(0)),
            value: loc(l),
        });
    }
    // int to float
    params.push(Param::buffer("f", E, BufferRole::Write, Some(1)));
    locals.push(LocalDecl::scalar(E));
    body.push(Stmt::Let {
        local: 7,
        init: ScalarExpr::Cast(
            Box::new(loc(2)),
            CastSpec {
                to: E,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            },
        ),
    });
    body.push(Stmt::Store {
        place: at(6, loc(0)),
        value: loc(7),
    });
    let k = KernelFn::new("casts", ElementKind::F64, params, locals, Block(body))
        .with_launch(lanes(32));
    let mut args = vec![f64s(&xs), i32s(&ints)];
    for t in tys {
        args.push(zeros(t, 64));
    }
    args.push(zeros(E, 64));
    check(
        "casts",
        &k,
        &args,
        32,
        &[
            &[("(l1) != (l1) ? (int)0", "0 ? (int)0")],
            &[("(l1) >= 2147483648.0 ?", "(l1) > 2147483648.0 ?")],
        ],
    );
}

#[test]
fn masked_load_store_select_neg_abs() {
    // x has 10 elements; threads past it are masked off (the load must touch nothing there).
    let x: Vec<f32> = vec![
        1.0,
        -2.0,
        f32::NAN,
        0.0,
        -0.0,
        5.5,
        -7.25,
        3.0,
        f32::INFINITY,
        -1.0,
    ];
    let k = KernelFn::new(
        "mask",
        E,
        vec![
            Param::buffer("x", E, BufferRole::Read, Some(1)),
            Param::buffer("y", E, BufferRole::ReadWrite, Some(1)),
            Param::buffer("z", E, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(E),
            LocalDecl::scalar(E),
            LocalDecl::scalar(E),
        ],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: tid(),
            },
            Stmt::MaskedLoad {
                local: 1,
                place: at(0, loc(0)),
                mask: bin(BinaryOp::CmpLt, loc(0), ku32(10)),
                fill: ScalarExpr::Const(-9.0),
            },
            Stmt::Let {
                local: 2,
                init: ScalarExpr::Unary(
                    UnaryOp::Neg,
                    Box::new(ScalarExpr::Unary(UnaryOp::Abs, Box::new(loc(1)))),
                ),
            },
            Stmt::Let {
                local: 3,
                init: ScalarExpr::Select(
                    Box::new(loc(1)),
                    Box::new(loc(2)),
                    Box::new(ScalarExpr::Const(4.0)),
                ),
            },
            Stmt::MaskedStore {
                place: at(1, loc(0)),
                value: loc(3),
                mask: bin(
                    BinaryOp::CmpEq,
                    ScalarExpr::Binary(
                        BinaryOp::Min,
                        Box::new(plus(loc(0), ku32(0))),
                        Box::new(ku32(63)),
                    ),
                    loc(0),
                ),
            },
            Stmt::Store {
                place: at(2, loc(0)),
                value: loc(2),
            },
        ]),
    )
    .with_launch(lanes(32));
    let args = [f32s(&x), f32s(&vec![100.0; 64]), zeros(E, 64)];
    check(
        "mask",
        &k,
        &args,
        32,
        &[&[(
            "? p0[(long long)(l0) * p0_s0] :",
            "? p0[(long long)(0u) * p0_s0] :",
        )]],
    );
}

/// A kernel with `ops.len()` collectives over the value `x[tid]`, each result stored at
/// `tid + k * block`.
fn collective_kernel(
    name: &str,
    vt: ElementKind,
    rt: ElementKind,
    n: u32,
    kinds: &[CollectiveKind],
) -> KernelFn {
    collective_kernel_s(name, vt, rt, n, 2, kinds)
}

/// [`collective_kernel`] over `subs` subgroups of `n` lanes.
fn collective_kernel_s(
    name: &str,
    vt: ElementKind,
    rt: ElementKind,
    n: u32,
    subs: u32,
    kinds: &[CollectiveKind],
) -> KernelFn {
    let block = subs * n;
    let mut locals = vec![LocalDecl::scalar(ElementKind::U32), LocalDecl::scalar(vt)];
    let mut body = vec![
        Stmt::Let {
            local: 0,
            init: tid(),
        },
        Stmt::Load {
            local: 1,
            place: at(0, loc(0)),
        },
    ];
    for (j, kind) in kinds.iter().enumerate() {
        let l = 2 + j as u32;
        locals.push(LocalDecl::scalar(rt));
        body.push(Stmt::Collective {
            kind: kind.clone(),
            value: loc(1),
            result: l,
            scope: unpopped::exec::Scope::Subgroup,
        });
        body.push(Stmt::Store {
            place: at(1, plus(loc(0), ku32(i64::from(block) * j as i64))),
            value: loc(l),
        });
    }
    KernelFn::new(
        name,
        vt,
        vec![
            Param::buffer("x", vt, BufferRole::Read, Some(1)),
            Param::buffer("y", rt, BufferRole::Write, Some(1)),
        ],
        locals,
        Block(body),
    )
    .with_launch(LaunchShape::new([block, 1, 1], [1, 1, 1]).with_subgroup(n, n))
}

fn shuffle_kinds(n: i64) -> Vec<CollectiveKind> {
    let mut v = Vec::new();
    for mode in [
        ShuffleMode::Idx,
        ShuffleMode::Up,
        ShuffleMode::Down,
        ShuffleMode::Xor,
    ] {
        for op in [0, 1, 3, n - 1, n, n + 1, -1, -3] {
            v.push(CollectiveKind::Shuffle {
                mode,
                operand: ki32(op),
            });
        }
    }
    v
}

#[test]
fn shuffle_edge_rules_at_wave_32_and_64() {
    for n in [32u32, 64] {
        let x: Vec<f32> = (0..2 * n).map(|i| i as f32 * 1.5 - 7.0).collect();
        // Every mode at every edge operand, in one kernel.
        let kinds = shuffle_kinds(i64::from(n));
        let k = collective_kernel("shuf", E, E, n, &kinds);
        let args = [f32s(&x), zeros(E, 2 * n as usize * kinds.len())];
        check(&format!("shuf{n}"), &k, &args, n, &[]);
        // One collective per kernel, so each mutant's anchor is unique.
        let one = |mode, op| {
            let k = collective_kernel(
                "shuf1",
                E,
                E,
                n,
                &[CollectiveKind::Shuffle {
                    mode,
                    operand: ki32(op),
                }],
            );
            let a = [f32s(&x), zeros(E, 2 * n as usize)];
            (k, a)
        };
        let nn = i64::from(n);
        let (k, a) = one(ShuffleMode::Up, 3);
        check(
            &format!("up{n}"),
            &k,
            &a,
            n,
            &[&[("lane0 - (", "lane0 + (")]],
        );
        let (k, a) = one(ShuffleMode::Down, 3);
        check(
            &format!("down{n}"),
            &k,
            &a,
            n,
            &[&[("lane0 + (", "lane0 - (")]],
        );
        let (k, a) = one(ShuffleMode::Xor, 5);
        check(
            &format!("xor{n}"),
            &k,
            &a,
            n,
            &[&[("lane0 ^ (", "lane0 & (")]],
        );
        let (k, a) = one(ShuffleMode::Idx, nn + 2);
        let (anchor, rep) = (
            format!("% {nn}LL) + {nn}LL) % {nn}LL"),
            format!("% {nn}LL) + {nn}LL) % 1000LL"),
        );
        check(
            &format!("idx{n}"),
            &k,
            &a,
            n,
            &[&[(anchor.as_str(), rep.as_str())]],
        );
        // An out-of-range Up (every lane reads itself): drop the range check and the host shim's
        // modulo wraps it instead, so the differential must see it.
        let (k, a) = one(ShuffleMode::Up, nn + 5);
        check(&format!("upoor{n}"), &k, &a, n, &[&[("src0 >= 0 && ", "")]]);
    }
}

#[test]
fn votes_all_any_ballot() {
    for n in [32u32, 64] {
        // Three subgroups: all non-zero, a mixed one, all zero.
        let x: Vec<i32> = (0..3 * n)
            .map(|i| match i / n {
                0 => 1 + (i % 3) as i32,
                1 => i32::from(i % 5 == 0),
                _ => 0,
            })
            .collect();
        let mut kinds = vec![
            CollectiveKind::Vote(VoteKind::All),
            CollectiveKind::Vote(VoteKind::Any),
        ];
        if n == 32 {
            kinds.push(CollectiveKind::Vote(VoteKind::Ballot));
        }
        let k = collective_kernel_s("vote", ElementKind::I32, ElementKind::U32, n, 3, &kinds);
        let args = [
            i32s(&x),
            zeros(ElementKind::U32, 3 * n as usize * kinds.len()),
        ];
        check(&format!("vote{n}"), &k, &args, n, &[]);
        let one = |kind| {
            let k = collective_kernel_s("vote1", ElementKind::I32, ElementKind::U32, n, 3, &[kind]);
            let a = [i32s(&x), zeros(ElementKind::U32, 3 * n as usize)];
            (k, a)
        };
        let (k, a) = one(CollectiveKind::Vote(VoteKind::All));
        check(
            &format!("all{n}"),
            &k,
            &a,
            n,
            &[&[("__all(v0 != 0)", "__any(v0 != 0)")]],
        );
        let (k, a) = one(CollectiveKind::Vote(VoteKind::Any));
        check(
            &format!("any{n}"),
            &k,
            &a,
            n,
            &[&[("__any(v0 != 0)", "__all(v0 != 0)")]],
        );
        if n == 32 {
            let (k, a) = one(CollectiveKind::Vote(VoteKind::Ballot));
            check(
                "ballot32",
                &k,
                &a,
                n,
                &[&[("(unsigned int)__ballot(", "~(unsigned int)__ballot(")]],
            );
        }
    }
}

#[test]
fn integer_reductions_and_broadcast() {
    use ElementKind::{I32, U32};
    for n in [32u32, 64] {
        let x: Vec<i32> = (0..2 * n)
            .map(|i| (i as i32 * 2_654_435 - 91_000) ^ (i as i32 * 7))
            .collect();
        for vt in [I32, U32] {
            let data = if vt == I32 {
                i32s(&x)
            } else {
                buf(
                    U32,
                    x.iter().flat_map(|v| (*v as u32).to_le_bytes()).collect(),
                )
            };
            let ops = [
                ReduceOp::Sum,
                ReduceOp::Max,
                ReduceOp::Min,
                ReduceOp::And,
                ReduceOp::Or,
                ReduceOp::Xor,
                ReduceOp::Prod,
            ];
            let kinds: Vec<CollectiveKind> =
                ops.iter().map(|o| CollectiveKind::Reduce(*o)).collect();
            let k = collective_kernel("red", vt, vt, n, &kinds);
            let args = [data.clone(), zeros(vt, 2 * n as usize * kinds.len())];
            check(&format!("red{n}_{vt:?}"), &k, &args, n, &[]);
        }
        // Mutants on single-collective kernels (unique anchors).
        let one = |op| {
            let k = collective_kernel("red1", I32, I32, n, &[CollectiveKind::Reduce(op)]);
            let a = [i32s(&x), zeros(I32, 2 * n as usize)];
            (k, a)
        };
        let (k, a) = one(ReduceOp::Sum);
        check(
            &format!("sum{n}"),
            &k,
            &a,
            n,
            &[
                &[("+ (unsigned int)t0", "- (unsigned int)t0")],
                &[("o0 >>= 1", "o0 >>= 2")],
            ],
        );
        let (k, a) = one(ReduceOp::Max);
        check(&format!("max{n}"), &k, &a, n, &[&[("r0 > t0", "r0 < t0")]]);
        // Broadcasts from constant lanes.
        let from = [0, 5, i64::from(n) - 1];
        let kinds: Vec<CollectiveKind> = from
            .iter()
            .map(|l| CollectiveKind::Broadcast { lane: ki32(*l) })
            .collect();
        let k = collective_kernel("bc", E, E, n, &kinds);
        let xf: Vec<f32> = (0..2 * n).map(|i| i as f32 * 0.5 + 1.0).collect();
        let args = [f32s(&xf), zeros(E, 2 * n as usize * kinds.len())];
        check(&format!("bc{n}"), &k, &args, n, &[]);
        let k = collective_kernel(
            "bc1",
            E,
            E,
            n,
            &[CollectiveKind::Broadcast { lane: ki32(5) }],
        );
        let args = [f32s(&xf), zeros(E, 2 * n as usize)];
        check(&format!("bc1_{n}"), &k, &args, n, &[&[(", 5, ", ", 6, ")]]);
    }
}

#[test]
fn shared_memory_with_a_block_barrier() {
    use unpopped::exec::{AllocDecl, Extent, MemorySpace, Scope};
    use unpopped::kernel::BarrierKind;
    let tile = |index| Place::Alloc { alloc: 0, index };
    let k = KernelFn::new(
        "rev",
        E,
        vec![
            Param::buffer("x", E, BufferRole::Read, Some(1)),
            Param::buffer("y", E, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(E),
            LocalDecl::scalar(E),
        ],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: tid(),
            },
            Stmt::Load {
                local: 1,
                place: at(0, loc(0)),
            },
            Stmt::Store {
                place: tile(loc(0)),
                value: loc(1),
            },
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            },
            Stmt::Load {
                local: 2,
                place: tile(ScalarExpr::Sub(Box::new(ku32(63)), Box::new(loc(0)))),
            },
            Stmt::Store {
                place: at(1, loc(0)),
                value: loc(2),
            },
        ]),
    )
    .with_allocs(vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        E,
        Extent::Static(64),
    )])
    .with_launch(lanes(32));
    let x: Vec<f32> = (0..64).map(|i| i as f32 * 3.0 - 11.0).collect();
    let args = [f32s(&x), zeros(E, 64)];
    // Mutant: the upper half stores late and the barrier is gone, so the lower half reads stale
    // shared memory (a 150 ms sleep makes the race certain on a host).
    check(
        "shared_rev",
        &k,
        &args,
        32,
        &[&[
            (
                "s0[(long long)(l0)] = l1;",
                "if (l0 >= 32) std::this_thread::sleep_for(std::chrono::milliseconds(150)); s0[(long long)(l0)] = l1;",
            ),
            ("__syncthreads();", ""),
        ]],
    );
}
