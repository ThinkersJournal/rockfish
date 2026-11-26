// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared by the tests that need a host `clang++`.

/// A host-clang test found no `clang++`: skip loudly, or FAIL when `REQUIRE_CLANG` is set.
///
/// CI sets `REQUIRE_CLANG=1` (and installs clang), so a runner without `clang++` cannot pass
/// these tests while testing nothing. A developer machine without clang still skips.
#[allow(
    dead_code,
    reason = "each integration test binary compiles this module and uses a different subset"
)]
pub fn skip_or_fail(msg: &str) {
    if std::env::var("REQUIRE_CLANG").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("REQUIRE_CLANG is set but {msg}");
    }
    eprintln!("{msg}");
}
