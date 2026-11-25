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
