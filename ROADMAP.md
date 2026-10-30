# Rockfish roadmap

Order matters: survey, then design, then code. Every code row ships with a negative control (a test that fails when the behaviour is removed). Linux and Windows from the start; anything not measured on real AMD hardware is labelled unmeasured (there is no ROCm test box yet).

| Row | What | Status |
|---|---|---|
| R0 | Workspace skeleton, CI (fmt, clippy, test x3 OS, doc, cargo-deny, portfolio), CODEOWNERS | in PR |
| R1 | Branch protection requiring the R0 checks (PM) | waits on R0 |
| R2 | `docs/SURVEY.md`: existing crates (rocm-rs, rocmrc, cubecl-hip-sys, rocm-sys), ROCm/HIP SDK library names per OS, what loads on Windows; options first, recommendation, no code | next |
| R3 | `docs/DESIGN.md`: sys/safe split, runtime loading, per-OS library table, error model, Fuel consumption (with the Fuel lane) | in PR |
| R3b | Spike: HIP version / struct-layout handling; result in `docs/SPIKE-R3b-versions.md` | in PR |
| R4a | `rockfish-core`: loader, per-OS name table, error type, tests against a real system library | in PR |
| R4b-1 | Vendor the HIP headers with provenance and a byte-for-byte check | in PR |
| R4b-2 | `rockfish-hip-sys` and `rockfish-hiprtc-sys`: bindgen-generated HIP runtime + hipRTC bindings on the R4a loader | in PR |
| R5 | `rockfish`: safe HIP layer (devices, streams, memory, modules, kernels) | after R4 |
| R6 | Math libraries: rocBLAS, hipBLASLt, rocFFT, rocRAND, rocSOLVER, rocSPARSE, MIOpen | after R5 |
| R7 | HIP C++ parser + emitter on Unpopped's `Frontend`/`Backend` contract (`Unpopped/docs/parser-emitter-ownership-map.md`, ROADMAP 6.7) | after R3 |
| R8 | Fuel ROCm backend built on Rockfish (design with the Fuel lane; Baracuda is the precedent) | after R5 |
