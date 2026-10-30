# Rockfish lane HANDOFF (new lane, created 2026-10-30 by the PM)

You own **Rockfish**: a comprehensive Rust interface to AMD ROCm and HIP, of the same caliber as Baracuda is for CUDA. Crates reserved on crates.io at 0.0.0: `rockfish`, `rockfish-sys` (owner ciresnave). Repo: ThinkersJournal/rockfish (this checkout, `C:\Projects\rockfish`).

CireSnave, 2026-10-30, verbatim: *"We need a lane for building our own ROCm crate of the same caliber as Baracuda is for CUDA."* and *"ROCm support on Windows is thinner than Linux but Windows *is supported*.  Should we add what ROCm support we can under Windows?"* (answer given: yes). *"I'd like to have comprehensive ROCm, and oneMKL support built for Fuel and for Unpopped so we can really test the limits of every manufacturer's software stack."* And: *"HIP C++ should be owned by Rockfish and Rockfish should write the parser and emitter for that."*

## Facts the PM measured (2026-10-30, crates.io/GitHub reads, nothing compiled)
rocm-rs 0.5.2 (MIT): HIP, rocBLAS, MIOpen, rocFFT, rocRAND, rocSOLVER, rocSPARSE, mostly raw bindings. rocmrc 0.5.1 (MIT/Apache): cudarc-style safe layer for HIP, hipRTC, rocBLAS, hipBLASLt, tiny. cubecl-hip-sys 7.14.x (Tracel, MIT/Apache, 1.9M downloads): bindgen for HIP runtime + hiprtc, runtime-loaded, Linux only. rocm-sys 0.4.1: one person, 147 downloads. hanzo-rocm looks like a copy of rocm-rs: skip. Recommended plan (CireSnave has not ruled on it): depend on cubecl-hip-sys for the HIP runtime layer on Linux or generate our own, generate our own bindings for rocBLAS, hipBLASLt, MIOpen, rocFFT, rocRAND, rocSOLVER, rocSPARSE, write the safe layer ourselves cudarc-style. **Verify each claim yourself before building on it.**

## Hard requirements
- Linux AND Windows from the start: libraries loaded at runtime, per-OS library names in one place; measure what loads on Windows (HIP SDK) and on which card.
- Hardware: CireSnave's desktop rig (dual boot Windows/Linux): RTX 4060, on a PEX switch a Radeon VII (gfx906), an Arc A770, a Tesla P40; a 32 GB MI50 (gfx906) is on order. The laptop has an AMD GPU too (identify it before relying on it). There is no ROCm test box yet: say so in every claim you cannot measure.
- Fuel (`C:\Projects\fuel`) has no ROCm backend; the eventual deliverable is a Fuel ROCm backend built on Rockfish (Baracuda is the precedent for how Fuel consumes a vendor crate). Design with the Fuel lane; do not edit its repo.
- Parser + emitter for HIP C++ on Unpopped's contract (`Frontend`, `Backend`/`Lowering`; read `C:\Projects\Unpopped\docs\parser-emitter-ownership-map.md`, ROADMAP 6.7). Unpopped holds neither.

(see COMMON rules below)
## Rules that bite here
Read `C:\Projects\CLAUDE.md` (loaded) and `C:\Projects\AGENT-ONBOARDING.md` once. Work in a worktree (`C:\Projects\<repo>-wt-<topic>`), never branch in the primary checkout. Version bump on every pushed change (pre-1.0 breaking = the SECOND number); the PM allocates the number at gate time. crates.io versions only, no path or git dependencies; all dependencies on their latest versions. Trace and verify every third-party file against its origin and keep its licence and credit (CireSnave: "I am not a plageurist"). Every claim verified on the forge or by measurement; every absence needs a positive control. Compact `[READY]`/`[STATUS]` messages, short. Peers change ID on every restart: `list_peers`, match on cwd. Restart at task boundaries (HANDOFF rewrite, `lane-restart assert-idle` last, then `--self`). This is an open-source project from day one: no secrets, no private paths in committed files.

## First deliverables (all lanes of this family)
0. `ROADMAP.md` with rows and statuses; a CI workflow (fmt, clippy, tests, cargo-deny, and the portfolio checks workflow the other repos use: copy the shape from `C:\Projects\capant\.github`) in your first PR; then `[ASK]` the PM to enable branch protection requiring those checks (CI and protection everywhere).
1. A survey and design note (options first, recommendation, no code) before any crate code.
