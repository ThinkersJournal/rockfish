//! Raw, runtime-loaded bindings to the AMD rocBLAS library.
//!
//! Like `rockfish-hip-sys`, nothing links against ROCm at build time, and a
//! missing library or entry point is an error value. The bindings in
//! `generated` come from `tools/gen-bindings` and the vendored rocBLAS headers
//! (see `THIRD-PARTY-NOTICES.md`); `hipStream_t` is the type from
//! `rockfish-hip-sys`, so a stream made there can be passed straight in.
//!
//! The headers are rocBLAS 5.2.0 (ROCm 7.2.4). The only library measured is
//! `rocblas.dll` from Ollama's ROCm 7.1 bundle on one Windows laptop: it exports
//! every function bound here. The Linux names are unverified guesses from the
//! soname convention, and nothing here has been run on a GPU yet.

use rockfish_core::LibrarySpec;

/// The rocBLAS library. Newest first.
pub const ROCBLAS: LibrarySpec = LibrarySpec {
    logical_name: "ROCBLAS",
    windows: &["rocblas.dll"],
    linux: &["librocblas.so.5", "librocblas.so.4", "librocblas.so"],
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
