// SPDX-License-Identifier: MIT OR Apache-2.0
//! R7-D15: Unpopped's opaque-tile statements (`TileLoad`, `TileFill`, `TileMma`, `TileStore`) lowered to three AMD matrix
//! atoms, as clang builtin calls over per-lane fragments.
//!
//! **Standing text (board 519 B), verbatim:** lifted by CireSnave 2026-11-26 (board 519 B) as his risk call; not a clearance; no
//! attorney; the Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read. The PM's chart read of the
//! AMD matrix patents (patents-2 #471, relayed 2026-11-28): the three atoms below are outside as read; no hold on this row. No
//! transpose option exists or is added (US12405770B2), and no convert-then-accumulate fusion is made: a `TileMma` is exactly
//! one builtin.
//!
//! **The three atoms** (and nothing else; every other shape, type pair or target is declined by name):
//!
//! | Atom (builtin) | Targets | Tiles `a x b -> c` |
//! |---|---|---|
//! | `__builtin_amdgcn_mfma_f32_16x16x4f32` | gfx908, gfx90a, gfx942 | f32 16x4 . 4x16 -> f32 16x16 |
//! | `__builtin_amdgcn_mfma_f64_16x16x4f64` | gfx90a, gfx942 | f64 16x4 . 4x16 -> f64 16x16 |
//! | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | gfx1100-1103, gfx1150-1153 | f16 16x16 . 16x16 -> f32 16x16 |
//!
//! **Lane maps** are the ones `docs/DESIGN-R7-MFMA.md` section 2a derives from AMD's MIT-licensed `matrix_calculator.py`
//! (pinned commit 94b1f16bf3a2) and the PM ran (coordinates in `tests/mfma_layout_reference.rs`); they are restated as
//! `Atom::coord` below. `lane` is HIP's `__lane_id()`:
//!
//! * MFMA f32 (wave 64): `A[i][k]` in lane `16k + i`; `B[k][j]` in lane `16k + j`; `D[4g + r][j]` in accumulator element `r` of
//!   lane `16g + j`.
//! * MFMA f64 (wave 64): A, B as f32; `D[4r + g][j]` in element `r` of lane `16g + j`.
//! * WMMA f16 (wave 32): `A[i][k]` is element `k` of lanes `i` and `i + 16`; `B[k][j]` is element `k` of lanes `j` and `j + 16`;
//!   `D[2r + g][j]` in element `r` of lane `16g + j`.
//!
//! **What is and is not verified.** The three builtin spellings and the instructions they select were checked by compiling a
//! call of each with clang 18 for the AMDGPU target (`-S`: `v_mfma_f32_16x16x4f32` and `v_mfma_f64_16x16x4f64` on gfx90a and
//! gfx942, `v_wmma_f32_16x16x16_f16` on gfx1100; gfx908 has no f64 MFMA, which is why that atom is gfx90a/gfx942 only); the
//! emitted kernels are re-checked the same way by `tests/tile_syntax.rs` where such a clang exists. **No lowered kernel has run
//! on a device**: this machine's Radeon 610M (gfx1036) has neither MFMA nor WMMA and there is no ROCm box, so every lane
//! map is UNMEASURED on silicon (they are the calculator's, not a device's). Hardware accumulation order and width differ from
//! the oracle's exact-products contract: a device result is compared within a tolerance, never by bytes.
//!
//! **Declined by name:** a tile that is not an operand of a `TileMma`; a tile used in two roles (A, B, accumulator); storing
//! an operand (A or B) tile; a launch that does not state the target's subgroup size; any shape, element pair or target not in the
//! table; and `lower_kernel` without a target.

use std::collections::BTreeMap;

use unpopped::exec::MemorySpace;
use unpopped::kernel::{Block, KernelFn, Stmt};
use unpopped_vocab::ElementKind;

/// One of the three lowered matrix atoms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Atom {
    MfmaF32,
    MfmaF64,
    WmmaF16,
}

/// Which operand of a `TileMma` a tile is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    A,
    B,
    /// Accumulator and result (`c` and `d`).
    C,
}

const MFMA_F32_TARGETS: [&str; 3] = ["gfx908", "gfx90a", "gfx942"];
const MFMA_F64_TARGETS: [&str; 2] = ["gfx90a", "gfx942"];
const WMMA_TARGETS: [&str; 8] = [
    "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153",
];

impl Atom {
    /// The clang builtin and its argument list for operands `a`, `b`, `c` (immediates `cbsz`, `abid`, `blgp` are 0).
    pub(crate) fn call(self, a: &str, b: &str, c: &str) -> String {
        match self {
            Self::MfmaF32 => {
                format!("__builtin_amdgcn_mfma_f32_16x16x4f32({a}, {b}, {c}, 0, 0, 0)")
            }
            Self::MfmaF64 => {
                format!("__builtin_amdgcn_mfma_f64_16x16x4f64({a}, {b}, {c}, 0, 0, 0)")
            }
            Self::WmmaF16 => format!("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32({a}, {b}, {c})"),
        }
    }

    pub(crate) const fn wave(self) -> u32 {
        match self {
            Self::MfmaF32 | Self::MfmaF64 => 64,
            Self::WmmaF16 => 32,
        }
    }

    /// The fragment's C type and element count for a role.
    pub(crate) const fn fragment(self, role: Role) -> (&'static str, usize) {
        match (self, role) {
            (Self::MfmaF32, Role::A | Role::B) => ("float", 1),
            (Self::MfmaF32, Role::C) => ("rf_v4f32", 4),
            (Self::MfmaF64, Role::A | Role::B) => ("double", 1),
            (Self::MfmaF64, Role::C) => ("rf_v4f64", 4),
            (Self::WmmaF16, Role::A | Role::B) => ("rf_v16f16", 16),
            (Self::WmmaF16, Role::C) => ("rf_v8f32", 8),
        }
    }

    /// The tile element `(row, column)`, as C expressions in `rf_lane`, held in fragment element `e` of this lane.
    pub(crate) fn coord(self, role: Role, e: usize) -> (String, String) {
        let lane16 = "(rf_lane % 16)".to_string();
        let grp = "(rf_lane / 16)".to_string();
        match (self, role) {
            (Self::MfmaF32 | Self::MfmaF64, Role::A) => (lane16, grp),
            (Self::MfmaF32 | Self::MfmaF64, Role::B) => (grp, lane16),
            (Self::MfmaF32, Role::C) => (format!("(4 * {grp} + {e})"), lane16),
            (Self::MfmaF64, Role::C) => (format!("(4 * {e} + {grp})"), lane16),
            (Self::WmmaF16, Role::A) => (lane16, e.to_string()),
            (Self::WmmaF16, Role::B) => (e.to_string(), lane16),
            (Self::WmmaF16, Role::C) => (format!("(2 * {e} + {grp})"), lane16),
        }
    }

    /// The `typedef` the atom's fragments need, if any (one line each).
    pub(crate) const fn typedefs(self) -> &'static [&'static str] {
        match self {
            Self::MfmaF32 => &["typedef float rf_v4f32 __attribute__((ext_vector_type(4)));"],
            Self::MfmaF64 => &["typedef double rf_v4f64 __attribute__((ext_vector_type(4)));"],
            Self::WmmaF16 => &[
                "typedef _Float16 rf_v16f16 __attribute__((ext_vector_type(16)));",
                "typedef float rf_v8f32 __attribute__((ext_vector_type(8)));",
            ],
        }
    }
}

/// What the tiles of one kernel lower to.
#[derive(Debug, Default)]
pub(crate) struct TilePlan {
    /// Per tile allocation: its atom and role.
    pub tiles: BTreeMap<u32, (Atom, Role)>,
}

/// The atom for one `TileMma` over `[d, a, b, c]`, or the reason there is none.
fn atom_of(k: &KernelFn, [d, a, b, c]: [u32; 4], arch: &str) -> Result<Atom, String> {
    let shape = |i: u32| {
        let al = k.allocs.get(i as usize);
        al.and_then(|x| x.tile.map(|t| (t.rows, t.cols, x.elem)))
            .ok_or_else(|| format!("tile {i} is not a Tile allocation"))
    };
    let (sa, sb, sc, sd) = (shape(a)?, shape(b)?, shape(c)?, shape(d)?);
    let what = format!(
        "a `TileMma` of {:?} {}x{} . {:?} {}x{} into {:?} {}x{} on `{arch}`",
        sa.2, sa.0, sa.1, sb.2, sb.0, sb.1, sc.2, sc.0, sc.1
    );
    let atoms = "only f32 16x4.4x16->16x16 (gfx908, gfx90a, gfx942), f64 the same (gfx90a, gfx942) and f16 16x16.16x16->f32 16x16 (gfx1100-1103, gfx1150-1153) are lowered";
    let on = |list: &[&str]| list.contains(&arch);
    let accum_ok = sc == sd;
    let atom = match (sa, sb, sc) {
        ((16, 4, ElementKind::F32), (4, 16, ElementKind::F32), (16, 16, ElementKind::F32))
            if accum_ok && on(&MFMA_F32_TARGETS) =>
        {
            Some(Atom::MfmaF32)
        }
        ((16, 4, ElementKind::F64), (4, 16, ElementKind::F64), (16, 16, ElementKind::F64))
            if accum_ok && on(&MFMA_F64_TARGETS) =>
        {
            Some(Atom::MfmaF64)
        }
        ((16, 16, ElementKind::F16), (16, 16, ElementKind::F16), (16, 16, ElementKind::F32))
            if accum_ok && on(&WMMA_TARGETS) =>
        {
            Some(Atom::WmmaF16)
        }
        _ => None,
    };
    atom.ok_or_else(|| format!("{what}: {atoms}"))
}

fn walk(
    k: &KernelFn,
    b: &Block,
    arch: &str,
    p: &mut TilePlan,
    used: &mut Vec<u32>,
) -> Result<(), String> {
    let set = |p: &mut TilePlan, tile: u32, atom: Atom, role: Role| match p.tiles.get(&tile) {
        Some(&(a0, r0)) if (a0, r0) != (atom, role) => Err(format!(
            "tile `{}` is used as {r0:?} and as {role:?} (a tile has one lane layout; no conversion between layouts is made)",
            k.allocs[tile as usize].name
        )),
        _ => {
            p.tiles.insert(tile, (atom, role));
            Ok(())
        }
    };
    for s in &b.0 {
        match s {
            Stmt::TileMma { d, a, b: bb, c } => {
                let atom = atom_of(k, [*d, *a, *bb, *c], arch)?;
                set(p, *a, atom, Role::A)?;
                set(p, *bb, atom, Role::B)?;
                set(p, *c, atom, Role::C)?;
                set(p, *d, atom, Role::C)?;
            }
            Stmt::TileLoad { d, .. } | Stmt::TileFill { d, .. } => used.push(*d),
            Stmt::TileStore { a, .. } => used.push(*a),
            Stmt::If { then, els, .. } => {
                walk(k, then, arch, p, used)?;
                walk(k, els, arch, p, used)?;
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => walk(k, body, arch, p, used)?,
            _ => {}
        }
    }
    Ok(())
}

/// Plan the tiles of `k` for target `arch` (wavefront `wave`), or say why it is declined.
pub(crate) fn plan(k: &KernelFn, wave: u32, arch: &str) -> Result<TilePlan, String> {
    let mut p = TilePlan::default();
    let mut used = Vec::new();
    walk(k, &k.body, arch, &mut p, &mut used)?;
    for t in used {
        if !p.tiles.contains_key(&t) {
            return Err(format!(
                "tile `{}` is no operand of a `TileMma`, so it has no AMD lane layout (declined by name)",
                k.allocs[t as usize].name
            ));
        }
    }
    for (t, (atom, _)) in &p.tiles {
        if atom.wave() != wave {
            return Err(format!(
                "tile `{}`: the atom needs wave size {} but target `{arch}` has {wave}",
                k.allocs[*t as usize].name,
                atom.wave()
            ));
        }
    }
    debug_assert!(
        p.tiles
            .keys()
            .all(|t| k.allocs[*t as usize].space == MemorySpace::Tile)
    );
    Ok(p)
}
