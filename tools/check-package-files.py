#!/usr/bin/env python3
# Copied from the AOCL lane's tools/check-package-files.py (same owner, same licence); only the default crate glob changed to rockfish*.
"""First-publish checklist: for every crate, list the files `cargo package` would ship and FAIL when a crate that carries a
provenance statement lacks one. A crate needs NOTICE in its package when its directory has a NOTICE, its sources point at the
clean-room tables (`docs/cleanroom/`), OR its README mentions NOTICE or provenance (wrappers too); every crate needs README.md and a LICENSE-* file. Run before EVERY first publish:
    python tools/check-package-files.py [crate ...]      # exit 1 on any problem; prints each package's NOTICE/README/LICENSE
Why: aocl-rng-sys 0.2.11 was published without its NOTICE (found on crates.io after publishing)."""
import glob, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
names = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(os.path.join(REPO, "crates", "rockfish*")) if os.path.isdir(d))
bad = 0
for name in names:
    d = os.path.join(REPO, "crates", name)
    r = subprocess.run(["cargo", "package", "--list", "--allow-dirty", "-p", name], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{name}: cargo package --list FAILED: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
        bad += 1
        continue
    files = r.stdout.split()
    srcs = ""
    for f in glob.glob(os.path.join(d, "src", "**", "*.rs"), recursive=True) + glob.glob(os.path.join(d, "README.md")):
        srcs += open(f, encoding="utf8", errors="replace").read()
    readme = open(os.path.join(d, "README.md"), encoding="utf8", errors="replace").read() if os.path.exists(os.path.join(d, "README.md")) else ""
    # A crate needs NOTICE in its package when its directory has one, its sources/README point at the clean-room tables, or
    # its README mentions NOTICE or provenance (wrappers too): a README that says "see NOTICE" must not dangle.
    needs_notice = (
        os.path.exists(os.path.join(d, "NOTICE"))
        or "docs/cleanroom/" in srcs
        or re.search(r"(?<![\w.-])NOTICE(?![\w.])", readme)  # our NOTICE, not AMD's NOTICE.txt
        or re.search(r"(?im)(\*\*provenance\b|^provenance\b)", readme)  # a Provenance statement, not a link to PROVENANCE-QUESTIONS.md
    )
    problems = []
    if "README.md" not in files:
        problems.append("README.md missing from the package")
    if not any(f.startswith("LICENSE") for f in files):
        problems.append("no LICENSE* file in the package")
    if needs_notice and "NOTICE" not in files:
        problems.append("NOTICE missing from the package (the crate has a provenance statement)" + ("" if os.path.exists(os.path.join(d, "NOTICE")) else ": the crate has no NOTICE file at all"))
    print(f"{name:30} {'OK ' if not problems else 'BAD'} NOTICE={'NOTICE' in files} README={'README.md' in files} LICENSE={[f for f in files if f.startswith('LICENSE')]}" + ("  " + "; ".join(problems) if problems else ""))
    bad += bool(problems)
sys.exit(1 if bad else 0)
