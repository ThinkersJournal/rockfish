// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Hip::lower_kernel` over Unpopped's published per-`Stmt`-variant fixtures (`fixtures` feature).
//!
//! **What this shows:** each of the 46 fixtures passes Unpopped's own checker (so the refusal is
//! the emitter's, not an invalid input), and `Hip` answers `UnsupportedKernel` for every one,
//! never a panic and never text. A variant on the declined list (`docs/PATENT-SWEEP.md`, board
//! 434 C / 506) is refused **by name**; `Collective`, which now includes the exclusive
//! Max/Min/Prod scans (Unpopped 0.58.4), is among them.
//! **What it does not show:** a lowering of any variant; there is none. Which statement arm a
//! fixture reaches first is recorded in `ARM_BY_NAME` below, by measurement at unpopped 0.59.3.

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::fixtures::{Fixture, fixtures};

fn refusal(name: &str, fx: &Fixture) -> String {
    let funcs = fx.functions();
    assert!(!funcs.is_empty(), "{name}: fixture has no function");
    let mut first = None;
    for f in funcs {
        match Hip.lower_kernel(f) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                first.get_or_insert(detail);
            }
            other => panic!("{name}: expected UnsupportedKernel, got {other:?}"),
        }
    }
    first.expect("at least one refusal")
}

#[test]
fn every_fixture_is_checker_valid() {
    let all = fixtures();
    assert_eq!(
        all.len(),
        46,
        "the published fixture set is one per Stmt variant"
    );
    for (name, fx) in &all {
        fx.check()
            .unwrap_or_else(|e| panic!("{name}: fixture refused by the checker: {e:?}"));
    }
}

#[test]
fn every_fixture_is_refused_as_unsupported_kernel() {
    for (name, fx) in fixtures() {
        let detail = refusal(name, &fx);
        assert!(detail.starts_with("hip backend"), "{name}: {detail}");
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

/// Prints, for the U62 answer, the arm each fixture reaches: declined-by-name, no-lowering, or the blanket no-kernel-layer refusal.
/// Run with `--nocapture`; asserts only that the three arms partition the set.
#[test]
fn arm_reached_per_fixture_is_by_name_for_every_variant() {
    let mut declined = 0;
    let mut unlowered = 0;
    let mut blanket = 0;
    for (name, fx) in fixtures() {
        let d = refusal(name, &fx);
        if d.contains("declined on AMD hardware") {
            declined += 1;
            eprintln!("DECLINED  {name}");
        } else if d.contains("no lowering for") {
            unlowered += 1;
            eprintln!("UNLOWERED {name}");
        } else if d.contains("does not lower kernel-layer kernels") {
            blanket += 1;
            eprintln!("BLANKET   {name}");
        } else {
            panic!("{name}: no known arm: {d}");
        }
    }
    eprintln!("declined={declined} unlowered={unlowered} blanket={blanket}");
    assert_eq!(declined + unlowered + blanket, 46);
}
