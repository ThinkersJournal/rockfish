//! Raw, runtime-loaded bindings to the AMD rocSPARSE library.
//!
//! Like `rockfish-hip-sys`, nothing links against ROCm at build time, and a
//! missing library or entry point is an error value. The bindings in
//! `generated` come from `tools/gen-bindings` and the vendored rocSPARSE
//! headers (see `THIRD-PARTY-NOTICES.md`). A curated set of single (`s`) and
//! double (`d`) precision functions is bound: handle and matrix-descriptor
//! management, format conversion, sparse level-1 (`axpyi`, `doti`, `gthr`,
//! `sctr`), `csrmv`, `csrmm`, `csrsv`, `csrilu0`, `csric0`, `csrgeam`, and the
//! generic `spmv`, `spmm`, `spgemm` and `sddmm` API with its descriptors. Not
//! bound yet: complex variants, the other storage formats' routines (BSR, ELL,
//! HYB, COO-specific), `spsv`/`spsm`, and the debugging entry points.
//!
//! The headers are rocSPARSE from ROCm 7.2.4. UNVERIFIED end to end: no
//! rocSPARSE library was found on the one machine this was written on (Ollama's
//! ROCm bundle ships rocBLAS and hipBLASLt, not rocSPARSE), so neither the
//! library names below (guesses from the soname and DLL conventions) nor the
//! exported symbols have been checked against a real install, and no sparse
//! kernel was run.

use rockfish_core::LibrarySpec;

/// The rocSPARSE library. Newest first.
pub const ROCSPARSE: LibrarySpec = LibrarySpec {
    logical_name: "ROCSPARSE",
    windows: &["rocsparse.dll", "librocsparse.dll"],
    linux: &["librocsparse.so.1", "librocsparse.so"],
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
    rustdoc::all
)]
#[rustfmt::skip]
mod generated;
pub use generated::*;
