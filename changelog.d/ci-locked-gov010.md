kind: ci

`cargo clippy`, `cargo test` and `cargo doc` run with `--locked` in CI (CodeRipper SUP-001). The 16 hand-written `#[allow]` attributes now carry a `reason =` (CodeRipper GOV-010; the 103 in generated bindings stay for a generator pass). No code change.

Every hand-written source file carries `SPDX-License-Identifier: MIT OR Apache-2.0`; the nine generated bindings carry their upstream licence (MIT, and BSD-2-Clause for rocSOLVER) and a derived-from line, written by `gen-bindings` (CodeRipper SUP-016). `THIRD-PARTY-NOTICES.md` corrected: rocSOLVER is BSD-2-Clause, not MIT.
