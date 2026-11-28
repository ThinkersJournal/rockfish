// SPDX-License-Identifier: MIT OR Apache-2.0
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
    rustdoc::all,
    reason = "bindgen output is not ours to restyle; the generated bindings trip these lints by construction (regenerate with `gen-bindings`)"
)]
#[rustfmt::skip]
mod generated;
pub use generated::*;
