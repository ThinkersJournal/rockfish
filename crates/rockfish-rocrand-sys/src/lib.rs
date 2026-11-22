//! Raw, runtime-loaded bindings to the AMD rocRAND library (the host API).
//!
//! Like `rockfish-hip-sys`, nothing links against ROCm at build time, and a
//! missing library or entry point is an error value. The bindings in
//! `generated` come from `tools/gen-bindings` and the vendored `rocrand.h`
//! (see `THIRD-PARTY-NOTICES.md`). Thirty functions are bound: generator
//! creation (device, host and blocking-host), seeding, offset, ordering and
//! quasi-random dimensions, the uniform, normal, log-normal, Poisson and raw
//! integer generators at single and double precision, the discrete and Poisson
//! distributions, and the Sobol direction-vector and scramble-constant tables.
//! Not bound: the three half-precision generators (`__half` is a HIP type this
//! workspace does not vendor), and the device-side headers.
//!
//! The headers are rocRAND from ROCm 7.2.4. UNVERIFIED end to end: no rocRAND
//! library was found on the one machine this was written on (Ollama's ROCm
//! bundle has none), so neither the library names below (guesses from the
//! soname and DLL conventions) nor the exported symbols have been checked
//! against a real install, and no random number was generated.

use rockfish_core::LibrarySpec;

/// The rocRAND library. Newest first.
pub const ROCRAND: LibrarySpec = LibrarySpec {
    logical_name: "ROCRAND",
    windows: &["rocrand.dll", "librocrand.dll"],
    linux: &["librocrand.so.1", "librocrand.so"],
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
