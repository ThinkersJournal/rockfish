// SPDX-License-Identifier: MIT OR Apache-2.0
//! Raw, runtime-loaded bindings to the AMD rocSOLVER library.
//!
//! Like `rockfish-hip-sys`, nothing links against ROCm at build time, and a
//! missing library or entry point is an error value. The bindings in
//! `generated` come from `tools/gen-bindings` and the vendored rocSOLVER
//! headers (see `THIRD-PARTY-NOTICES.md`). A curated set of 36 functions is
//! bound: the single (`s`) and double (`d`) precision dense factorisations and
//! solvers (`potrf`, `potrs`, `potri`, `getrf`, `getrs`, `getri`, `geqrf`,
//! `gelqf`, `orgqr`, `ormqr`, `gesv`, `posv`, `gels`, `gesvd`, `syev`, `syevd`,
//! `sygv`, `trtri`). Not bound yet: complex and batched variants, the aliases,
//! and the logging entry points.
//!
//! rocSOLVER sits on rocBLAS: its handle, status, fill, operation, side,
//! diagonal and integer types are those of `rockfish-rocblas-sys`.
//!
//! The headers are rocSOLVER from ROCm 7.2.4. UNVERIFIED end to end: no
//! rocSOLVER library was found on the one machine this was written on (Ollama's
//! ROCm bundle ships rocBLAS and hipBLASLt, not rocSOLVER), so neither the
//! library names below (guesses from the soname and DLL conventions) nor the
//! exported symbols have been checked against a real install, and no
//! factorisation was run.

use rockfish_core::LibrarySpec;

/// The rocSOLVER library. Newest first.
pub const ROCSOLVER: LibrarySpec = LibrarySpec {
    logical_name: "ROCSOLVER",
    windows: &["rocsolver.dll", "librocsolver.dll"],
    linux: &["librocsolver.so.0", "librocsolver.so"],
};

#[allow(
    clippy::all,
    dead_code,
    missing_docs,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unnecessary_transmutes,
    unsafe_op_in_unsafe_fn,
    deref_nullptr,
    rustdoc::all,
    reason = "bindgen output is not ours to restyle; the generated bindings trip these lints by construction (regenerate with `gen-bindings`)"
)]
#[rustfmt::skip]
mod generated;
pub use generated::*;
