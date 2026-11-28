// SPDX-License-Identifier: MIT OR Apache-2.0
//! Raw, runtime-loaded bindings to the AMD rocFFT library.
//!
//! Like `rockfish-hip-sys`, nothing links against ROCm at build time, and a
//! missing library or entry point is an error value. The bindings in
//! `generated` come from `tools/gen-bindings` and the vendored `rocfft.h` (see
//! `THIRD-PARTY-NOTICES.md`). The public C API is bound (29 functions). Excluded: `rocfft_execution_info_set_mode` and
//! `rocfft_execution_info_get_events` (inside `#if 0` in the header) and
//! `rocfft_get_offline_tuner_handle` (only under `ROCFFT_BUILD_OFFLINE_TUNER`). rocFFT
//! takes the stream and events as `void*` (a `hipStream_t` is passed as is), so
//! this crate has no dependency on `rockfish-hip-sys`.
//!
//! The header is rocFFT from ROCm 7.2.4. UNVERIFIED end to end: no rocFFT
//! library was found on the one machine this was written on (Ollama's ROCm
//! bundle ships rocBLAS and hipBLASLt, not rocFFT), so neither the library
//! names below (guesses from the soname and DLL conventions) nor the exported
//! symbols have been checked against a real install, and no plan was executed.
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

/// The rocFFT library. Newest first.
pub const ROCFFT: LibrarySpec = LibrarySpec {
    logical_name: "ROCFFT",
    windows: &["rocfft.dll", "librocfft.dll"],
    linux: &["librocfft.so.0", "librocfft.so"],
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
