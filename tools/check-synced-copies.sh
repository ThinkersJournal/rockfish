#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# A published .crate does not contain the repo root, so every crate dir carries synced copies: LICENSE-MIT (all crates),
# THIRD-PARTY-NOTICES.md (each crate with a generated.rs: the root file's header plus only the section for the headers that
# crate was generated from) and rocsolver's BSD-2-Clause text. Fails when one differs or is missing. LICENSE-APACHE copies
# follow in a later PR (size); this script checks them only where present.
set -u
bad=0
python3 - <<'PY' || bad=1
import re, os, sys
s = open('THIRD-PARTY-NOTICES.md', encoding='utf-8', newline='').read()
head, *secs = re.split(r'(?m)^(?=## )', s)
# Only the vendored-header sections are synced into crates; other sections (e.g. the calculator notice, not vendored) are skipped.
by = {m.group(1): x for x in secs if (m := re.match(r'## AMD .+? \(`vendor/(\w+)/`\)', x))}
want = {'hip-sys': 'hip', 'hiprtc-sys': 'hip', 'rocblas-sys': 'rocblas', 'hipblaslt-sys': 'hipblaslt', 'miopen-sys': 'miopen',
        'rocfft-sys': 'rocfft', 'rocrand-sys': 'rocrand', 'rocsparse-sys': 'rocsparse', 'rocsolver-sys': 'rocsolver'}
bad = 0
for c in sorted(os.listdir('crates')):
    d = f'crates/{c}/'
    has_gen = os.path.exists(d + 'src/generated.rs')
    p = d + 'THIRD-PARTY-NOTICES.md'
    if has_gen:
        v = want.get(c.replace('rockfish-', ''))
        if v is None: print('UNMAPPED generated crate', c); bad = 1; continue
        exp = (head + by[v].rstrip('\r\n') + ('\r\n' if '\r\n' in s else '\n')).replace('\r\n', '\n')
        got = open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n') if os.path.exists(p) else None
        if got != exp: print('DIFF/MISSING', p); bad = 1
    elif os.path.exists(p): print('STRAY', p); bad = 1
sys.exit(bad)
PY
for d in crates/*/; do
  cmp -s LICENSE-MIT "${d}LICENSE-MIT" || { echo "DIFF/MISSING ${d}LICENSE-MIT"; bad=1; }
  [ -f "${d}LICENSE-APACHE" ] && { cmp -s LICENSE-APACHE "${d}LICENSE-APACHE" || { echo "DIFF ${d}LICENSE-APACHE"; bad=1; }; }
done
cmp -s vendor/rocsolver/LICENSE.md crates/rockfish-rocsolver-sys/LICENSE-BSD-2-Clause || { echo "DIFF/MISSING rocsolver BSD text"; bad=1; }
echo "checked $(ls -d crates/*/ | wc -l) crates; exit $bad"
exit $bad
