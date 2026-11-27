// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Hip::lower_kernel_for`: the subgroup collectives CireSnave's board 506 "A" (2026-11-26) lifted.
//!
//! **What this shows:** on a stated RDNA (wave32) or CDNA (wave64) target, a subgroup-scope
//! shuffle, vote, reduction or broadcast lowers to HIP's `__shfl`/`__shfl_xor`/`__all`/`__any`/
//! `__ballot`, every shuffle source lane computed in range by the emitter (the IR's edge rules,
//! spelled, never inherited); the text parses against a Rockfish-written shim. Scans, block-scope
//! collectives, a `Ballot` on a 64-lane wavefront, GCN3, Vega, an unlisted target, and a launch
//! that does not state the subgroup size are each refused by name; the plain `lower_kernel`
//! (no target) still refuses every `Collective`. **What it does not show:** that a device runs it:
//! UNMEASURED (no ROCm box), nothing was executed, and the edge-lane results are the emitted
//! arithmetic, not a measurement. Patent note: basis board 506 and docs/PATENT-SWEEP.md
//! US8539204B2 claims 1, 12 and 13, dated 2026-11-26; not a clearance.

use std::path::PathBuf;
use std::process::Command;

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind,
};
use unpopped_vocab::{ElementKind, TargetId};

mod common;

fn t(arch: &str) -> TargetId {
    TargetId::parse(&format!("rocm:{arch}")).unwrap()
}
fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::I32)
}

/// `out[tid] = collective(in[tid])` with a subgroup of `n` lanes stated by the launch.
fn kernel(
    kind: CollectiveKind,
    scope: Scope,
    vt: ElementKind,
    rt: ElementKind,
    n: u32,
) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(unpopped::exec::ExecId::ThreadIdx(0))],
    };
    KernelFn::new(
        "coll",
        vt,
        vec![
            Param::buffer("in", vt, BufferRole::Read, Some(1)),
            Param::buffer("out", rt, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(vt), LocalDecl::scalar(rt)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Collective {
                kind,
                value: ScalarExpr::Local(0),
                result: 1,
                scope,
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([n * 4, 1, 1], [1, 1, 1]).with_subgroup(n, n))
}

fn ok(k: &KernelFn, arch: &str) -> String {
    Hip.lower_kernel_for(k, &t(arch))
        .unwrap_or_else(|e| panic!("{arch}: {e:?}"))
        .source
}
fn refusal(k: &KernelFn, arch: &str) -> String {
    match Hip.lower_kernel_for(k, &t(arch)) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("{arch}: expected UnsupportedKernel, got {other:?}"),
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
        PathBuf::from("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang++.exe"),
    ]
    .into_iter()
    .find(|c| Command::new(c).arg("--version").output().is_ok_and(|o| o.status.success()))
}

fn parses(label: &str, src: &str) {
    let Some(cc) = clang() else {
        common::skip_or_fail("no clang++ found");
        return;
    };
    let dir = std::env::temp_dir().join("rockfish-subgroup");
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
        "{label}:\n{}\n{src}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn shuffle(mode: ShuffleMode, op: i64) -> CollectiveKind {
    CollectiveKind::Shuffle {
        mode,
        operand: ci(op),
    }
}

#[test]
fn shuffles_spell_the_edge_rules_in_range() {
    use ElementKind::F32;
    for (arch, n) in [("gfx1100", 32u32), ("gfx90a", 64)] {
        for (mode, tag, expect) in [
            (ShuffleMode::Idx, "idx", "% %nLL) + %nLL) % %nLL"),
            (ShuffleMode::Up, "up", "- ("),
            (ShuffleMode::Down, "down", "+ ("),
            (ShuffleMode::Xor, "xor", "^ ("),
        ] {
            let s = ok(
                &kernel(shuffle(mode, 3), Scope::Subgroup, F32, F32, n),
                arch,
            );
            let expect = expect.replace("%n", &n.to_string());
            assert!(s.contains(&expect), "{arch} {tag}: {s}");
            // The source lane is range-checked, with the lane's own value otherwise.
            assert!(s.contains(&format!("< {n}LL) ?")), "{arch} {tag}: {s}");
            assert!(
                s.contains(&format!(", {n})")),
                "{arch} {tag}: width {n}: {s}"
            );
            assert!(s.contains("__shfl(v0"), "{s}");
            parses(&format!("shuf_{arch}_{tag}"), &s);
        }
    }
}

#[test]
fn votes_reduce_and_broadcast_lower() {
    use ElementKind::{I32, U32};
    let s = ok(
        &kernel(
            CollectiveKind::Vote(VoteKind::Any),
            Scope::Subgroup,
            I32,
            U32,
            64,
        ),
        "gfx942",
    );
    assert!(s.contains("__any(v0 != 0)"), "{s}");
    parses("any64", &s);
    let s = ok(
        &kernel(
            CollectiveKind::Vote(VoteKind::All),
            Scope::Subgroup,
            I32,
            U32,
            32,
        ),
        "gfx1030",
    );
    assert!(s.contains("__all(v0 != 0)"), "{s}");
    let s = ok(
        &kernel(
            CollectiveKind::Vote(VoteKind::Ballot),
            Scope::Subgroup,
            I32,
            U32,
            32,
        ),
        "gfx1201",
    );
    assert!(s.contains("(unsigned int)__ballot(v0 != 0)"), "{s}");
    parses("ballot32", &s);
    for (op, needle) in [
        (ReduceOp::Sum, "(int)((unsigned int)r0 + (unsigned int)t0)"),
        (ReduceOp::Max, "r0 > t0"),
        (ReduceOp::Xor, "r0 ^ t0"),
    ] {
        let s = ok(
            &kernel(CollectiveKind::Reduce(op), Scope::Subgroup, I32, I32, 64),
            "gfx908",
        );
        assert!(s.contains(needle), "{needle}: {s}");
        assert!(s.contains("o0 = 64 / 2"), "{s}");
        assert!(s.contains("__shfl_xor(r0, o0, 64)"), "{s}");
        parses(
            &format!("red_{needle:.4}").replace(|c: char| !c.is_ascii_alphanumeric(), "_"),
            &s,
        );
    }
    let s = ok(
        &kernel(
            CollectiveKind::Broadcast { lane: ci(5) },
            Scope::Subgroup,
            ElementKind::F64,
            ElementKind::F64,
            32,
        ),
        "gfx1100",
    );
    assert!(s.contains("__shfl(v0, 5, 32)"), "{s}");
    parses("bcast", &s);
}

#[test]
fn what_stays_declined_is_declined_by_name() {
    use ElementKind::{F32, I32, U32};
    let scan = CollectiveKind::Scan {
        op: ReduceOp::Sum,
        exclusive: false,
    };
    let d = refusal(
        &kernel(scan.clone(), Scope::Subgroup, I32, I32, 32),
        "gfx1100",
    );
    assert!(
        d.contains("`Collective`") && d.contains("US8539204B2") && d.contains("434 C"),
        "{d}"
    );
    let d = refusal(
        &kernel(
            CollectiveKind::Reduce(ReduceOp::Sum),
            Scope::Block,
            I32,
            I32,
            32,
        ),
        "gfx1100",
    );
    assert!(
        d.contains("`Collective`") && d.contains("US8539204B2"),
        "{d}"
    );
    // Ballot over 64 lanes: refused, not silently truncated.
    let d = refusal(
        &kernel(
            CollectiveKind::Vote(VoteKind::Ballot),
            Scope::Subgroup,
            I32,
            U32,
            64,
        ),
        "gfx90a",
    );
    assert!(d.contains("Ballot") && d.contains("32"), "{d}");
    // A float reduction: order not decided.
    let d = refusal(
        &kernel(
            CollectiveKind::Reduce(ReduceOp::Sum),
            Scope::Subgroup,
            F32,
            F32,
            32,
        ),
        "gfx1100",
    );
    assert!(d.contains("float reduction"), "{d}");
    // Targets: GCN3, Vega, unlisted, non-rocm.
    let ok_k = kernel(shuffle(ShuffleMode::Xor, 1), Scope::Subgroup, F32, F32, 64);
    for arch in ["gfx906", "gfx900", "gfx803"] {
        let d = refusal(&ok_k, arch);
        assert!(d.contains("GCN3 and Vega"), "{arch}: {d}");
    }
    assert!(refusal(&ok_k, "gfx9999").contains("declined by name"));
    // The launch must state the wavefront: a 32-lane launch on a 64-lane target.
    let wrong = kernel(shuffle(ShuffleMode::Xor, 1), Scope::Subgroup, F32, F32, 32);
    assert!(refusal(&wrong, "gfx90a").contains("subgroup size 64"));
    // Control (the same kernel, the right size, lowers): the refusals above are not vacuous.
    assert!(
        ok(
            &kernel(shuffle(ShuffleMode::Xor, 1), Scope::Subgroup, F32, F32, 64),
            "gfx90a"
        )
        .contains("__shfl")
    );
    // Plain `lower_kernel` states no target: every Collective stays declined.
    match Hip.lower_kernel(&kernel(
        shuffle(ShuffleMode::Xor, 1),
        Scope::Subgroup,
        F32,
        F32,
        64,
    )) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("US8539204B2"), "{detail}")
        }
        other => panic!("{other:?}"),
    }
}
