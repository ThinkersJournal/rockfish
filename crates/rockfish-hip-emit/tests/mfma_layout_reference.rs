// SPDX-License-Identifier: MIT OR Apache-2.0
//! A pure-Rust reference for the CDNA `v_mfma_f32_16x16x4f32` lane layout (docs/DESIGN-R7-MFMA.md, 2a), checked against
//! Unpopped's `TileMma` oracle (products exact, ascending `k`, one rounding to `f32`).
//!
//! **What this shows:** the lane maps hand-derived from AMD's `matrix_calculator.py` (MIT, pinned commit 94b1f16bf3a2,
//! `InstCalcGfx9`) are internally consistent: each of A, B, D covers its registers/lanes exactly once, and a model of the
//! instruction built *only from lane positions* (lane `16g + r` of A holds `A[r][g]`; of B holds `B[g][r]`; the product pairs
//! A lane `16g + i` with B lane `16g + j`) reproduces the oracle's `D` after the output map gathers it. A wrong lane in a map
//! (the mutant in `a_swapped_lane_is_caught`) breaks that.
//! The output map `d_slot` is checked by value too: the model's lanes compute their own element from the inverse form of the
//! formula (`D[4*(l/16)+r][l%16]` in register `r`, lane `l`), and `d_slot` gathers D; a swapped output slot changes a value
//! (`a_swapped_output_slot_is_caught`). The inverse form restates the same calculator formula, so this tests that `d_slot` and
//! its inverse agree, not that AMD's hardware does.
//! **What it does not show:** that hardware behaves this way. The model is ours, derived by hand from the calculator's code
//! and never run against it or against a device (no ROCm box); the lane-pairing rule is the calculator's own formula
//! restated, so agreement is not independent evidence about AMD hardware. UNVERIFIED. Nothing here lowers an MFMA.

use unpopped::exec::{AllocDecl, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::ElementKind;

const M: usize = 16;
const N: usize = 16;
const K: usize = 4;
const LANES: usize = 64;

/// `A[i][k]` -> lane (calculator `__get_input_reg_lanes`, M=16, K=4, B=1: `lane = k*M*B + i`); one register.
fn a_lane(i: usize, k: usize) -> usize {
    16 * k + i
}
/// `B[k][j]` -> lane (same formula with `j` as the outer index).
fn b_lane(k: usize, j: usize) -> usize {
    16 * k + j
}
/// `D[i][j]` -> (register, lane) (calculator `__get_output_reg_lanes`, 32-bit: `multirow_height = 4`).
fn d_slot(i: usize, j: usize) -> (usize, usize) {
    (i % 4, 16 * (i / 4) + j)
}

fn kernel() -> KernelFn {
    use ElementKind::F32;
    let rd = |n: &str| Param::buffer(n, F32, BufferRole::Read, Some(1));
    let at = |b: u32| Place::Logical {
        buffer: b,
        coords: vec![ScalarExpr::ConstInt(0, ElementKind::I32)],
    };
    let u = |v: i64| ScalarExpr::ConstInt(v, ElementKind::I32);
    let mut k = KernelFn::new(
        "mfma_ref",
        F32,
        vec![
            rd("a"),
            rd("b"),
            Param::buffer("d", F32, BufferRole::ReadWrite, Some(1)),
        ],
        Vec::<LocalDecl>::new(),
        Block(vec![
            Stmt::TileLoad {
                d: 0,
                src: at(0),
                elements_per_row: u(K as i64),
                origin: [u(0), u(0)],
            },
            Stmt::TileLoad {
                d: 1,
                src: at(1),
                elements_per_row: u(N as i64),
                origin: [u(0), u(0)],
            },
            Stmt::TileFill {
                d: 2,
                value: ScalarExpr::Const(0.0),
            },
            Stmt::TileMma {
                d: 2,
                a: 0,
                b: 1,
                c: 2,
            },
            Stmt::TileStore {
                a: 2,
                dst: at(2),
                elements_per_row: u(N as i64),
                origin: [u(0), u(0)],
            },
        ]),
    );
    k = k.with_launch(
        LaunchShape::new([LANES as u32, 1, 1], [1, 1, 1]).with_subgroup(LANES as u32, LANES as u32),
    );
    k.with_allocs(vec![
        AllocDecl::tile("a", F32, M as u32, K as u32),
        AllocDecl::tile("b", F32, K as u32, N as u32),
        AllocDecl::tile("d", F32, M as u32, N as u32),
    ])
}

fn buf(v: &[f32]) -> KernelArg {
    let bytes = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(
        ElementKind::F32,
        vec![v.len() as i64],
        vec![1],
        bytes,
    ))
}

fn oracle_d(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut args = vec![buf(a), buf(b), buf(&[0.0; M * N])];
    run_launch_with_subgroup(&kernel(), &mut args, Some(LANES as u32))
        .expect("oracle runs the tile kernel");
    match &args[2] {
        KernelArg::Buffer(t) => t
            .raw_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => unreachable!(),
    }
}

/// The instruction modelled from lane positions only. `amap` places `A[i][k]` into a lane of the A register.
fn model_d(
    a: &[f32],
    b: &[f32],
    amap: &dyn Fn(usize, usize) -> usize,
    dmap: &dyn Fn(usize, usize) -> (usize, usize),
) -> Vec<f32> {
    let (mut ra, mut rb) = ([0f32; LANES], [0f32; LANES]);
    for i in 0..M {
        for k in 0..K {
            ra[amap(i, k)] = a[i * K + k];
        }
    }
    for k in 0..K {
        for j in 0..N {
            rb[b_lane(k, j)] = b[k * N + j];
        }
    }
    // Lanes pair within a group of 16 by output position: D[i][j] = sum over groups g of A_lane[16g+i] * B_lane[16g+j],
    // accumulated in f64, ascending g (= k), rounded once to f32.
    // The hardware side is written from lane positions, as the inverse of `d_slot` (register `r`, lane `l` holds
    // `D[4*(l/16) + r][l % 16]`): each lane computes its own output element, then the map UNDER TEST gathers D by (i, j).
    let mut regs = [[0f32; LANES]; 4];
    for (r, reg) in regs.iter_mut().enumerate() {
        for (l, slot) in reg.iter_mut().enumerate() {
            let (i, j) = (4 * (l / 16) + r, l % 16);
            let s: f64 = (0..K)
                .map(|g| f64::from(ra[16 * g + i]) * f64::from(rb[16 * g + j]))
                .sum();
            *slot = s as f32;
        }
    }
    let mut d = vec![0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let (r, l) = dmap(i, j);
            d[i * N + j] = regs[r][l];
        }
    }
    d
}

fn inputs() -> (Vec<f32>, Vec<f32>) {
    // Distinct small integers: exact in f32, so equality is exact and a misplaced lane cannot cancel by symmetry.
    let a = (0..M * K).map(|x| (x as f32) * 0.5 - 7.0).collect();
    let b = (0..K * N).map(|x| ((x * 7 % 23) as f32) - 11.0).collect();
    (a, b)
}

#[test]
fn maps_cover_every_slot_exactly_once() {
    let mut a = vec![0; LANES];
    let mut b = vec![0; LANES];
    let mut d = vec![0; 4 * LANES];
    for i in 0..M {
        for k in 0..K {
            a[a_lane(i, k)] += 1;
        }
    }
    for k in 0..K {
        for j in 0..N {
            b[b_lane(k, j)] += 1;
        }
    }
    for i in 0..M {
        for j in 0..N {
            let (r, l) = d_slot(i, j);
            assert!(r < 4 && l < LANES);
            d[r * LANES + l] += 1;
        }
    }
    assert!(a.iter().all(|&c| c == 1), "A: 64 entries, one per lane");
    assert!(b.iter().all(|&c| c == 1), "B: 64 entries, one per lane");
    assert!(
        d.iter().all(|&c| c == 1),
        "D: 256 entries fill 4 registers x 64 lanes"
    );
}

#[test]
fn the_lane_model_matches_the_tile_oracle() {
    let (a, b) = inputs();
    let want = oracle_d(&a, &b);
    assert_eq!(model_d(&a, &b, &a_lane, &d_slot), want);
    // Control: the oracle result is not all zeros and not symmetric, so a match says something.
    assert!(want.iter().any(|&x| x != 0.0) && want[1] != want[N]);
}

#[test]
fn a_swapped_lane_is_caught() {
    let (a, b) = inputs();
    let want = oracle_d(&a, &b);
    // Mutant: lanes 0 and 1 of A swapped (i = 0 and 1 at k = 0).
    let swapped = |i: usize, k: usize| match a_lane(i, k) {
        0 => 1,
        1 => 0,
        l => l,
    };
    assert_ne!(
        model_d(&a, &b, &swapped, &d_slot),
        want,
        "the swap must change D"
    );
}

#[test]
fn a_swapped_output_slot_is_caught() {
    let (a, b) = inputs();
    let want = oracle_d(&a, &b);
    // Mutant: the slots of D[0][0] and D[0][1] swapped.
    let swapped = |i: usize, j: usize| match (i, j) {
        (0, 0) => d_slot(0, 1),
        (0, 1) => d_slot(0, 0),
        _ => d_slot(i, j),
    };
    assert_ne!(
        model_d(&a, &b, &a_lane, &swapped),
        want,
        "the swap must change D"
    );
}
