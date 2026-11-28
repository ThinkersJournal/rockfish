// SPDX-License-Identifier: MIT OR Apache-2.0
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
//!
//! # Enum typedef signedness
//!
//! The signedness of each C enum typedef in `generated.rs` follows the C compiler of the generation host:
//! the file is generated on Linux (where ROCm ships), so an enum with no negative enumerator is `c_uint` there
//! (MSVC would give `c_int`). The C ABI is the same 32-bit integer for these enums on MSVC and GCC, so the
//! call ABI is unchanged. Hand-written code names the generated typedef (`hipError_t`, `hiprtcResult`, ...),
//! never `c_int`/`c_uint`, so it does not depend on that choice. A Windows build with the Linux-generated file
//! is UNMEASURED beyond the loader tests run on the Windows laptop (see ROADMAP R6-A2). An enum with a negative
//! enumerator is `c_int` on both hosts; the HIP/hipRTC enums have none (grep of the generated file).

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
    rustdoc::all,
    reason = "bindgen output is not ours to restyle; the generated bindings trip these lints by construction (regenerate with `gen-bindings`)"
)]
#[rustfmt::skip]
mod generated;
pub use generated::*;
