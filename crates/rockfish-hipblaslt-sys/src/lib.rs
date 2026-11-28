// SPDX-License-Identifier: MIT OR Apache-2.0
//! Raw, runtime-loaded bindings to the AMD hipBLASLt library.
//!
//! Like `rockfish-hip-sys`, nothing links against ROCm at build time, and a
//! missing library or entry point is an error value. The bindings in
//! `generated` come from `tools/gen-bindings` and the vendored hipBLASLt and
//! hipblas-common headers (see `THIRD-PARTY-NOTICES.md`); `hipStream_t` is the
//! type from `rockfish-hip-sys`. Only the C API in `hipblaslt.h` is bound, not
//! the C++ extension API.
//!
//! The headers are hipBLASLt 1.2.2 (ROCm 7.2.4). The only library measured is
//! `libhipblaslt.dll` (sic, with the `lib` prefix) from Ollama's ROCm 7.1 bundle
//! on one Windows laptop: it exports every function bound here. The Linux names
//! are unverified guesses from the soname convention, and nothing here has been
//! run on a GPU.
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

/// The hipBLASLt library. Newest first.
pub const HIPBLASLT: LibrarySpec = LibrarySpec {
    logical_name: "HIPBLASLT",
    windows: &["libhipblaslt.dll", "hipblaslt.dll"],
    linux: &["libhipblaslt.so.1", "libhipblaslt.so.0", "libhipblaslt.so"],
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
