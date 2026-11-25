kind: ci

`cargo clippy`, `cargo test` and `cargo doc` run with `--locked` in CI (CodeRipper SUP-001). The 16 hand-written `#[allow]` attributes now carry a `reason =` (CodeRipper GOV-010; the 103 in generated bindings stay for a generator pass). No code change.
