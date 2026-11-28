// SPDX-License-Identifier: MIT OR Apache-2.0
//! R7-D15: `TileLoad`/`TileFill`/`TileMma`/`TileStore` lowered to the three AMD matrix atoms (`src/tile.rs`).
//!
//! **Standing text (board 519 B):** lifted by CireSnave 2026-11-26 (board 519 B) as his risk call; not a clearance; no
//! attorney; the Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read.
//!
//! **What this shows:** (1) one emitted-text snapshot per atom (`tests/fixtures/tile_*.cpp`; `BLESS=1` rewrites them);
//! (2) the lane maps are read back *from the emitted text* (a small evaluator over the C index expressions) and compared with
//! the coordinates the PM ran through AMD's calculator (the same points as `tests/mfma_layout_reference.rs`), with every slot
//! covered exactly once; (3) for f32 the maps read back from the text drive the same lane model as `mfma_layout_reference.rs`
//! (D recovered from lane positions only) and reproduce a plain matrix product; a lane swapped in the emitted maps is caught
//! by both the PM coordinates and the model; (4) every other shape, type pair, target and use is declined by name.
//! **What it does not show:** that any lowered kernel runs: no device was used (this machine's Radeon 610M, gfx1036, has neither
//! MFMA nor WMMA, and there is no ROCm box), so the maps are the calculator's, not silicon's: UNMEASURED. Whether clang accepts the
//! text is `tests/tile_syntax.rs`.

use rockfish_hip_emit::Hip;
use unpopped::backend::LowerError;
use unpopped::exec::{AllocDecl, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::{ElementKind, TargetId};

fn at(b: u32) -> Place {
    Place::Logical {
        buffer: b,
        coords: vec![ScalarExpr::ConstInt(0, ElementKind::I32)],
    }
}
fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::I32)
}

/// `d = a . b` over one tile each: load A, load B, zero C, `TileMma`, store D. `(m, k, n)` and the element types are free so
/// the declines can reuse it.
#[allow(clippy::too_many_arguments)]
fn kernel_with(
    name: &str,
    ab: ElementKind,
    acc: ElementKind,
    (m, k, n): (u32, u32, u32),
    wave: u32,
    compute: ElementKind,
    mutate: impl FnOnce(&mut Vec<Stmt>),
) -> KernelFn {
    let rd = |nm: &str| Param::buffer(nm, ab, BufferRole::Read, Some(1));
    let mut body = vec![
        Stmt::TileLoad {
            d: 0,
            src: at(0),
            elements_per_row: u(i64::from(k)),
            origin: [u(0), u(0)],
        },
        Stmt::TileLoad {
            d: 1,
            src: at(1),
            elements_per_row: u(i64::from(n)),
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
            elements_per_row: u(i64::from(n)),
            origin: [u(0), u(0)],
        },
    ];
    mutate(&mut body);
    KernelFn::new(
        name,
        compute,
        vec![
            rd("a"),
            rd("b"),
            Param::buffer("d", acc, BufferRole::ReadWrite, Some(1)),
        ],
        Vec::<LocalDecl>::new(),
        Block(body),
    )
    .with_launch(LaunchShape::new([wave, 1, 1], [1, 1, 1]).with_subgroup(wave, wave))
    .with_allocs(vec![
        AllocDecl::tile("a", ab, m, k),
        AllocDecl::tile("b", ab, k, n),
        AllocDecl::tile("d", acc, m, n),
    ])
}

use ElementKind::{F16, F32, F64};

fn mfma_f32() -> KernelFn {
    kernel_with("tile_mfma_f32", F32, F32, (16, 4, 16), 64, F32, |_| {})
}
fn mfma_f64() -> KernelFn {
    kernel_with("tile_mfma_f64", F64, F64, (16, 4, 16), 64, F64, |_| {})
}
fn wmma_f16() -> KernelFn {
    kernel_with("tile_wmma_f16", F16, F32, (16, 16, 16), 32, F32, |_| {})
}

fn lower(k: &KernelFn, arch: &str) -> Result<String, String> {
    let t = TargetId::parse(&format!("rocm:{arch}")).unwrap();
    match Hip.lower_kernel_for(k, &t) {
        Ok(g) => Ok(g.source),
        Err(LowerError::UnsupportedKernel { detail }) => Err(detail),
        Err(e) => panic!("expected Ok or UnsupportedKernel, got {e:?}"),
    }
}

fn norm(s: &str) -> String {
    s.replace("\r\n", "\n")
}

fn snapshot(file: &str, got: &str) {
    let path = format!("{}/tests/fixtures/{file}", env!("CARGO_MANIFEST_DIR"));
    if std::env::var("BLESS").is_ok() {
        std::fs::create_dir_all(format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))).unwrap();
        std::fs::write(&path, got).unwrap();
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e} (BLESS=1 writes it)"));
    assert_eq!(norm(&want), norm(got), "{file} differs (BLESS=1 rewrites)");
}

#[test]
fn snapshot_mfma_f32_gfx90a() {
    snapshot("tile_mfma_f32.cpp", &lower(&mfma_f32(), "gfx90a").unwrap());
}
#[test]
fn snapshot_mfma_f64_gfx90a() {
    snapshot("tile_mfma_f64.cpp", &lower(&mfma_f64(), "gfx90a").unwrap());
}
#[test]
fn snapshot_wmma_f16_gfx1100() {
    snapshot("tile_wmma_f16.cpp", &lower(&wmma_f16(), "gfx1100").unwrap());
}

/// The builtin is the atom's, once, and nothing else spells a matrix instruction.
#[test]
fn each_atom_emits_exactly_its_builtin() {
    let cases = [
        (
            mfma_f32(),
            "gfx908",
            "__builtin_amdgcn_mfma_f32_16x16x4f32(t0, t1, t2, 0, 0, 0)",
        ),
        (
            mfma_f32(),
            "gfx942",
            "__builtin_amdgcn_mfma_f32_16x16x4f32(t0, t1, t2, 0, 0, 0)",
        ),
        (
            mfma_f64(),
            "gfx942",
            "__builtin_amdgcn_mfma_f64_16x16x4f64(t0, t1, t2, 0, 0, 0)",
        ),
        (
            wmma_f16(),
            "gfx1103",
            "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(t0, t1, t2)",
        ),
        (
            wmma_f16(),
            "gfx1151",
            "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(t0, t1, t2)",
        ),
    ];
    for (k, arch, call) in cases {
        let src = lower(&k, arch).unwrap_or_else(|e| panic!("{arch}: {e}"));
        assert_eq!(src.matches("__builtin_amdgcn_").count(), 1, "{arch}\n{src}");
        assert!(src.contains(&format!("t2 = {call};")), "{arch}\n{src}");
    }
}

// ---- the lane maps, read back from the emitted text -------------------------------------------------------------------

/// A tiny evaluator for the emitted index expressions: integers, `rf_lane`, `+ - * / %` and parentheses (C precedence, integer
/// division, operands non-negative).
fn arith(expr: &str, lane: i64) -> i64 {
    struct P<'a> {
        s: &'a [u8],
        i: usize,
        lane: i64,
    }
    impl P<'_> {
        fn ws(&mut self) {
            while self.i < self.s.len() && self.s[self.i] == b' ' {
                self.i += 1;
            }
        }
        fn atom(&mut self) -> i64 {
            self.ws();
            if self.s[self.i] == b'(' {
                self.i += 1;
                let v = self.sum();
                self.ws();
                assert_eq!(self.s[self.i], b')');
                self.i += 1;
                return v;
            }
            if self.s[self.i..].starts_with(b"rf_lane") {
                self.i += 7;
                return self.lane;
            }
            let st = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            std::str::from_utf8(&self.s[st..self.i])
                .unwrap()
                .parse()
                .expect("a number")
        }
        fn prod(&mut self) -> i64 {
            let mut v = self.atom();
            loop {
                self.ws();
                match self.s.get(self.i) {
                    Some(b'*') => {
                        self.i += 1;
                        v *= self.atom();
                    }
                    Some(b'/') => {
                        self.i += 1;
                        v /= self.atom();
                    }
                    Some(b'%') => {
                        self.i += 1;
                        v %= self.atom();
                    }
                    _ => return v,
                }
            }
        }
        fn sum(&mut self) -> i64 {
            let mut v = self.prod();
            loop {
                self.ws();
                match self.s.get(self.i) {
                    Some(b'+') => {
                        self.i += 1;
                        v += self.prod();
                    }
                    Some(b'-') => {
                        self.i += 1;
                        v -= self.prod();
                    }
                    _ => return v,
                }
            }
        }
    }
    let mut p = P {
        s: expr.as_bytes(),
        i: 0,
        lane,
    };
    let v = p.sum();
    assert_eq!(p.i, expr.len(), "trailing text in `{expr}`");
    v
}

/// One memory access of the emitted text: fragment variable, element, and the tile `(row, column)` as expressions.
struct Access {
    tile: u32,
    elem: usize,
    row: String,
    col: String,
}

/// Every `t<N>[e] = p<M>[((long long)(0) + (rf_o0 + ROW) * rf_epr + rf_o1 + COL) * ...]` (a load) or the reverse (a store).
fn accesses(src: &str, store: bool) -> Vec<Access> {
    let mut out = Vec::new();
    for line in src.lines().map(str::trim) {
        let (frag, mem) = match line.split_once(" = ") {
            Some((l, r)) if !store && l.starts_with('t') && r.starts_with('p') => (l, r),
            Some((l, r)) if store && l.starts_with('p') && r.starts_with('t') => (r, l),
            _ => continue,
        };
        let frag = frag.trim_end_matches(';');
        let (tile, elem) = match frag[1..].split_once('[') {
            Some((t, e)) => (t.parse().unwrap(), e.trim_end_matches(']').parse().unwrap()),
            None => (frag[1..].parse().unwrap(), 0),
        };
        let key = "(rf_o0 + ";
        let a = mem.find(key).expect("row marker") + key.len();
        // ROW is a balanced expression ending at the `)` that closes `(rf_o0 + `.
        let mut depth = 1;
        let mut j = a;
        for (n, ch) in mem[a..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        j = a + n;
                        break;
                    }
                }
                _ => {}
            }
        }
        let row = mem[a..j].to_string();
        let rest = &mem[j..];
        let c0 = rest.find("rf_o1 + ").expect("col marker") + "rf_o1 + ".len();
        // COL runs to the `)` that closes the whole index (balanced).
        let mut depth = 0;
        let mut end = rest.len();
        for (n, ch) in rest[c0..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    if depth == 0 {
                        end = c0 + n;
                        break;
                    }
                    depth -= 1;
                }
                _ => {}
            }
        }
        out.push(Access {
            tile,
            elem,
            row,
            col: rest[c0..end].to_string(),
        });
    }
    out
}

/// For a tile: the `(lane, elem) -> (row, col)` map (`lane_map`) read from its emitted accesses.
fn slots(
    acc: &[Access],
    tile: u32,
    lanes: i64,
    lane_of: &dyn Fn(i64) -> i64,
) -> Vec<(i64, usize, i64, i64)> {
    let mut v = Vec::new();
    for a in acc.iter().filter(|a| a.tile == tile) {
        for l in 0..lanes {
            v.push((
                l,
                a.elem,
                arith(&a.row, lane_of(l)),
                arith(&a.col, lane_of(l)),
            ));
        }
    }
    v
}

fn ident(l: i64) -> i64 {
    l
}

/// `(row, col) -> every (lane, elem)` holding it.
fn holders(s: &[(i64, usize, i64, i64)], r: i64, c: i64) -> Vec<(i64, usize)> {
    s.iter()
        .filter(|x| x.2 == r && x.3 == c)
        .map(|x| (x.0, x.1))
        .collect()
}

/// The PM's executed calculator points (`tests/mfma_layout_reference.rs`, 2026-11-27) against the maps in the emitted text.
fn pm_points_hold(src: &str, which: &str, lane_of: &dyn Fn(i64) -> i64) -> bool {
    let loads = accesses(src, false);
    let (lanes, a, b, d) = match which {
        "f32" | "f64" => (
            64,
            slots(&loads, 0, 64, lane_of),
            slots(&loads, 1, 64, lane_of),
            slots(&accesses(src, true), 2, 64, lane_of),
        ),
        _ => (
            32,
            slots(&loads, 0, 32, lane_of),
            slots(&loads, 1, 32, lane_of),
            slots(&accesses(src, true), 2, 32, lane_of),
        ),
    };
    let _ = lanes;
    match which {
        "f32" => {
            holders(&a, 3, 2) == [(35, 0)]
                && holders(&b, 1, 5) == [(21, 0)]
                && holders(&d, 6, 3) == [(19, 2)]
                && holders(&d, 13, 7) == [(55, 1)]
        }
        "f64" => {
            holders(&d, 6, 3) == [(35, 1)]
                && holders(&a, 6, 3) == [(54, 0)]
                && holders(&a, 3, 2) == [(35, 0)]
        }
        _ => {
            // RDNA3 wave32: D[5][3] in element 2, lane 19; A[3][9] is element 9 of lanes 3 and 19.
            holders(&d, 5, 3) == [(19, 2)] && {
                let mut h = holders(&a, 3, 9);
                h.sort_unstable();
                h == [(3, 9), (19, 9)]
            }
        }
    }
}

#[test]
fn emitted_maps_match_the_pm_run_coordinates() {
    assert!(pm_points_hold(
        &lower(&mfma_f32(), "gfx90a").unwrap(),
        "f32",
        &ident
    ));
    assert!(pm_points_hold(
        &lower(&mfma_f64(), "gfx90a").unwrap(),
        "f64",
        &ident
    ));
    assert!(pm_points_hold(
        &lower(&wmma_f16(), "gfx1100").unwrap(),
        "wmma",
        &ident
    ));
}

#[test]
fn a_lane_swap_in_the_emitted_maps_is_caught_by_the_pm_coordinates() {
    // Mutant: lanes 19 and 21 exchanged in every map (the points above live in those lanes).
    let swap = |l: i64| match l {
        19 => 21,
        21 => 19,
        x => x,
    };
    assert!(!pm_points_hold(
        &lower(&mfma_f32(), "gfx90a").unwrap(),
        "f32",
        &swap
    ));
    // Control: the same predicate holds on the unmutated maps (the test above), so the failure is the swap's.
    assert!(pm_points_hold(
        &lower(&mfma_f32(), "gfx90a").unwrap(),
        "f32",
        &ident
    ));
    let swap_wmma = |l: i64| match l {
        3 => 19,
        19 => 3,
        x => x,
    };
    // For WMMA the A points are symmetric under 3 <-> 19; the D point lives in lane 19 alone and moves.
    assert!(!pm_points_hold(
        &lower(&wmma_f16(), "gfx1100").unwrap(),
        "wmma",
        &swap_wmma
    ));
}

#[test]
fn every_slot_is_covered_exactly_once() {
    let check = |src: &str, lanes: i64, elems_a: usize, elems_c: usize, a_copies: usize| {
        let loads = accesses(src, false);
        let a = slots(&loads, 0, lanes, &ident);
        let b = slots(&loads, 1, lanes, &ident);
        let d = slots(&accesses(src, true), 2, lanes, &ident);
        let (m, k) = (16i64, if elems_a == 16 { 16 } else { 4 });
        for i in 0..m {
            for kk in 0..k {
                assert_eq!(holders(&a, i, kk).len(), a_copies, "A[{i}][{kk}]");
                assert_eq!(holders(&b, kk, i).len(), a_copies, "B[{kk}][{i}]");
            }
        }
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(holders(&d, i, j).len(), 1, "D[{i}][{j}]");
            }
        }
        assert_eq!(d.len() as i64, 16 * 16);
        assert_eq!(d.len(), (lanes as usize) * elems_c);
    };
    check(&lower(&mfma_f32(), "gfx90a").unwrap(), 64, 1, 4, 1);
    check(&lower(&mfma_f64(), "gfx90a").unwrap(), 64, 1, 4, 1);
    // WMMA wave32: A and B sit in two lanes (`i` and `i + 16`): two holders per element.
    check(&lower(&wmma_f16(), "gfx1100").unwrap(), 32, 16, 8, 2);
}

// The f32 lane model of `tests/mfma_layout_reference.rs`: the instruction written from lane positions only, fed by the maps
// read from the emitted text, then compared with a plain product (inputs are small integers, exact in f32).
fn model_d_f32(src: &str, lane_of: &dyn Fn(i64) -> i64, a: &[f32], b: &[f32]) -> Vec<f32> {
    let loads = accesses(src, false);
    let (sa, sb) = (slots(&loads, 0, 64, lane_of), slots(&loads, 1, 64, lane_of));
    let sd = slots(&accesses(src, true), 2, 64, lane_of);
    let (mut ra, mut rb) = ([0f32; 64], [0f32; 64]);
    for &(l, _, r, c) in &sa {
        ra[l as usize] = a[(r * 4 + c) as usize];
    }
    for &(l, _, r, c) in &sb {
        rb[l as usize] = b[(r * 16 + c) as usize];
    }
    // The instruction: accumulator lane `l`, register `e` holds the output `(i, j)` the emitted store map puts there; the
    // lanes pair A lane `16g + i` with B lane `16g + j` (the calculator's rule, `mfma_layout_reference.rs`).
    let mut d = vec![0f32; 256];
    for &(l, _e, i, j) in &sd {
        let _ = l;
        let s: f64 = (0..4usize)
            .map(|g| f64::from(ra[16 * g + i as usize]) * f64::from(rb[16 * g + j as usize]))
            .sum();
        d[(i * 16 + j) as usize] = s as f32;
    }
    d
}

fn plain_d(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut d = vec![0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            d[i * 16 + j] = (0..4)
                .map(|k| f64::from(a[i * 4 + k]) * f64::from(b[k * 16 + j]))
                .sum::<f64>() as f32;
        }
    }
    d
}

#[test]
fn the_emitted_f32_maps_reproduce_the_product_and_a_swapped_lane_does_not() {
    let a: Vec<f32> = (0..64).map(|x| (x as f32) * 0.5 - 7.0).collect();
    let b: Vec<f32> = (0..64).map(|x| ((x * 7 % 23) as f32) - 11.0).collect();
    let src = lower(&mfma_f32(), "gfx90a").unwrap();
    let want = plain_d(&a, &b);
    assert_eq!(model_d_f32(&src, &ident, &a, &b), want);
    assert!(
        want.iter().any(|&x| x != 0.0) && want[1] != want[16],
        "control: the product is not trivial"
    );
    // Mutant: lanes 0 and 1 swapped in every emitted map (anchors: lane 0 holds A[0][0], so the swap moves a value).
    let swap = |l: i64| match l {
        0 => 1,
        1 => 0,
        x => x,
    };
    assert_ne!(
        model_d_f32(&src, &swap, &a, &b),
        want,
        "the swap must change D"
    );
}

// ---- declines by name ------------------------------------------------------------------------------------------------

#[test]
fn other_targets_shapes_and_types_are_declined_by_name() {
    for (k, arch, needle) in [
        (mfma_f32(), "gfx950", "`TileMma` of F32"),
        (mfma_f32(), "gfx906", "declined by name"),
        (mfma_f32(), "gfx1100", "`TileMma` of F32"),
        (mfma_f64(), "gfx908", "`TileMma` of F64"),
        (wmma_f16(), "gfx90a", "`TileMma` of F16"),
        (wmma_f16(), "gfx1030", "`TileMma` of F16"),
        (wmma_f16(), "gfx1200", "`TileMma` of F16"),
        // 32x32x2 f32 (a real MFMA shape, not an atom here).
        (
            kernel_with("s", F32, F32, (32, 2, 32), 64, F32, |_| {}),
            "gfx90a",
            "`TileMma` of F32",
        ),
        // bf16 in, f32 out.
        (
            kernel_with("s", ElementKind::Bf16, F32, (16, 16, 16), 32, F32, |_| {}),
            "gfx1100",
            "`TileMma` of Bf16",
        ),
    ] {
        let d = lower(&k, arch).expect_err(&format!("{arch} must be declined"));
        assert!(d.contains(needle), "{arch}: {d}");
    }
}

#[test]
fn no_target_is_declined_by_name() {
    use unpopped::backend::Backend;
    let e = Hip.lower_kernel(&mfma_f32()).expect_err("no target");
    let LowerError::UnsupportedKernel { detail } = e else {
        panic!("{e:?}")
    };
    assert!(detail.contains("Tile"), "{detail}");
}

#[test]
fn a_launch_without_the_subgroup_size_is_declined() {
    let k = mfma_f32().with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]));
    let d = lower(&k, "gfx90a").expect_err("no subgroup size stated");
    assert!(d.contains("subgroup size 64"), "{d}");
    let k = mfma_f32().with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    assert!(lower(&k, "gfx90a").is_err());
}

#[test]
fn operand_tile_stores_and_unused_tiles_are_declined() {
    // Store the A tile.
    let k = kernel_with("s", F32, F32, (16, 4, 16), 64, F32, |b| {
        b.push(Stmt::TileStore {
            a: 0,
            dst: at(2),
            elements_per_row: u(4),
            origin: [u(0), u(0)],
        });
    });
    // The checker wants `d` writable and of A's element type: it is (F32), so the refusal is the emitter's.
    let d = lower(&k, "gfx90a").expect_err("operand store");
    assert!(d.contains("`TileStore` of operand tile"), "{d}");
    // A tile that no `TileMma` uses.
    let mut k = kernel_with("s", F32, F32, (16, 4, 16), 64, F32, |b| {
        b.push(Stmt::TileFill {
            d: 3,
            value: ScalarExpr::Const(1.0),
        });
    });
    k.allocs.push(AllocDecl::tile("x", F32, 16, 16));
    let d = lower(&k, "gfx90a").expect_err("unused tile");
    assert!(d.contains("no operand of a `TileMma`"), "{d}");
    // A tile used as A and as B of one `TileMma` (the f16 16x16 tile is legal in both places for the checker).
    let k = kernel_with("s", F16, F32, (16, 16, 16), 32, F32, |b| {
        b[3] = Stmt::TileMma {
            d: 2,
            a: 0,
            b: 0,
            c: 2,
        };
    });
    let d = lower(&k, "gfx1100").expect_err("two roles");
    assert!(d.contains("is used as"), "{d}");
}

#[test]
fn f16_is_not_a_general_local_type() {
    // Control for the tile-only `_Float16`: a plain f16 buffer load into an f16 local is still declined.
    use unpopped::kernel::LocalDecl;
    let k = KernelFn::new(
        "f16_plain",
        F32,
        vec![Param::buffer("a", F16, BufferRole::Read, Some(1))],
        vec![LocalDecl::scalar(F16)],
        Block(vec![Stmt::Load {
            local: 0,
            place: at(0),
        }]),
    );
    let t = TargetId::parse("rocm:gfx90a").unwrap();
    assert!(Hip.lower_kernel_for(&k, &t).is_err());
}
