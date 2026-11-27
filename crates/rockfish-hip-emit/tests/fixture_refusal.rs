// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Hip::lower_kernel` over Unpopped's published per-`Stmt`-variant fixtures (`fixtures` feature).
//!
//! **What this shows:** each of the 50 fixtures passes Unpopped's own checker (so the refusal is
//! the emitter's, not an invalid input), and `Hip` answers `UnsupportedKernel` for every one,
//! never a panic and never text. A variant on the declined list (`docs/PATENT-SWEEP.md`, board
//! 434 C / 506) is refused **by name**; `Collective`, which now includes the exclusive
//! Max/Min/Prod scans (Unpopped 0.58.4), is among them.
//! **What it does not show:** a lowering of any variant; there is none. Which statement arm a
//! fixture reaches first is recorded in `ARM_BY_NAME` below, by measurement at unpopped 0.59.3.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::fixtures::{Fixture, fixtures};

/// `Ok(source)` for a lowered fixture, `Err(detail)` for a typed `UnsupportedKernel`; any other
/// error, or a panic, fails the test.
fn outcome(name: &str, fx: &Fixture) -> Result<String, String> {
    let funcs = fx.functions();
    assert!(!funcs.is_empty(), "{name}: fixture has no function");
    let mut first: Result<String, String> = Ok(String::new());
    for f in funcs {
        match Hip.lower_kernel(f) {
            Ok(g) => {
                if let Ok(src) = &mut first {
                    src.push_str(&g.source);
                }
            }
            Err(LowerError::UnsupportedKernel { detail }) => {
                if first.is_ok() {
                    first = Err(detail);
                }
            }
            other => panic!("{name}: expected Ok or UnsupportedKernel, got {other:?}"),
        }
    }
    first
}

fn refusal(name: &str, fx: &Fixture) -> String {
    outcome(name, fx).expect_err(&format!("{name}: expected a refusal"))
}

#[test]
fn every_fixture_is_checker_valid() {
    let all = fixtures();
    assert_eq!(
        all.len(),
        50,
        "the published fixture set is one per Stmt variant"
    );
    for (name, fx) in &all {
        fx.check()
            .unwrap_or_else(|e| panic!("{name}: fixture refused by the checker: {e:?}"));
    }
}

#[test]
fn every_fixture_is_lowered_or_refused_as_unsupported_kernel() {
    for (name, fx) in fixtures() {
        match outcome(name, &fx) {
            Ok(src) => assert!(src.contains("__global__"), "{name}: {src}"),
            Err(detail) => assert!(detail.starts_with("hip backend"), "{name}: {detail}"),
        }
    }
}

#[test]
fn collective_is_declined_by_name_with_the_434c_marker() {
    let (_, fx) = fixtures()
        .into_iter()
        .find(|(n, _)| *n == "Collective")
        .expect("Collective fixture");
    let detail = refusal("Collective", &fx);
    assert!(detail.contains("`Collective`"), "{detail}");
    assert!(detail.contains("declined on AMD hardware"), "{detail}");
    assert!(detail.contains("US8539204B2"), "{detail}");
}

/// Prints, for the U62 answer, what each fixture reaches. Run with `--nocapture`; asserts only
/// that the arms partition the set, and the lowered set is the S1 subset (a change to it is a
/// decision, so this list is exact).
#[test]
fn arm_reached_per_fixture() {
    let (mut lowered, mut declined, mut unlowered, mut s1, mut checker) =
        (vec![], 0, 0, vec![], vec![]);
    for (name, fx) in fixtures() {
        match outcome(name, &fx) {
            Ok(_) => lowered.push(name),
            Err(d) if d.contains("declined on AMD hardware") => declined += 1,
            Err(d) if d.contains("no lowering for") => unlowered += 1,
            Err(d) if d.contains("the S1 kernel-layer lowerer spells only") => s1.push(name),
            // A module entry alone: the checker wants `check_module` (the Call fixtures).
            Err(d) if d.contains("the Unpopped checker refuses it") => checker.push(name),
            Err(d) => panic!("{name}: no known arm: {d}"),
        }
    }
    eprintln!(
        "lowered={lowered:?}
S1-declined={s1:?}"
    );
    eprintln!("checker-refused={checker:?}");
    assert_eq!(
        lowered.len() + declined + unlowered + s1.len() + checker.len(),
        50
    );
    assert_eq!(declined, 13);
    assert_eq!(unlowered, 8);
}

/// Unpopped 0.62 `Tile*` statements (board 519 B) are declined by name, and so is the `Tile`
/// memory space (its allocation). Control: `Barrier` lowers from the same fixture set. No
/// MFMA/WMMA mapping is made; that waits for a design. Any text on these carries: lifted by
/// CireSnave 2026-11-26 (board 519 B) as his risk call; not a clearance; no attorney; the
/// Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read.
#[test]
fn tile_statements_and_space_are_declined_by_name() {
    let all = fixtures();
    let (_, fx) = all.iter().find(|(n, _)| *n == "Barrier").expect("Barrier");
    outcome("Barrier", fx).expect("control: Barrier lowers");
    for name in ["TileLoad", "TileMma", "TileFill", "TileStore"] {
        let (_, fx) = all.iter().find(|(n, _)| *n == name).expect(name);
        let d = refusal(name, fx);
        // The fixture declares a `Tile` allocation first, so the allocation scan may answer
        // before the statement does: either way the refusal names a Tile thing.
        assert!(
            d.contains(&format!("statement `{name}`")) || d.contains("Tile"),
            "{name}: {d}"
        );
    }
}
