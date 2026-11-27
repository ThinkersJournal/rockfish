#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Every crate dir carries synced copies of the root LICENSE-MIT/LICENSE-APACHE, and every crate with a generated.rs a copy of
# THIRD-PARTY-NOTICES.md (a published .crate does not contain the repo root). Fails when a copy differs or is missing.
set -u
bad=0
for d in crates/*/; do
  for f in LICENSE-MIT LICENSE-APACHE; do cmp -s "$f" "$d$f" || { echo "DIFF/MISSING $d$f"; bad=1; }; done
  if [ -f "${d}src/generated.rs" ]; then cmp -s THIRD-PARTY-NOTICES.md "${d}THIRD-PARTY-NOTICES.md" || { echo "DIFF/MISSING ${d}THIRD-PARTY-NOTICES.md"; bad=1; }; fi
done
cmp -s vendor/rocsolver/LICENSE.md crates/rockfish-rocsolver-sys/LICENSE-BSD-2-Clause || { echo "DIFF/MISSING rocsolver BSD text"; bad=1; }
echo "checked $(ls -d crates/*/ | wc -l) crates; exit $bad"
exit $bad
