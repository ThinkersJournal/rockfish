//! Raw, runtime-loaded bindings to the AMD MIOpen library (the HIP backend).
//!
//! Like `rockfish-hip-sys`, nothing links against ROCm at build time, and a
//! missing library or entry point is an error value. The bindings in
//! `generated` come from `tools/gen-bindings` and the vendored `miopen.h` (see
//! `THIRD-PARTY-NOTICES.md`); `hipStream_t` is the type from `rockfish-hip-sys`.
//! Only the handle, tensor, convolution and activation functions are bound,
//! which is the part Fuel's cuDNN-backed convolution needs.
//!
//! The header is MIOpen 3.5.1 (ROCm 7.2.4). UNVERIFIED end to end: no MIOpen
//! library was found on the one machine this was written on, so neither the
//! library names below (guesses from the soname and DLL conventions) nor the
//! exported symbols have been checked against a real install.

use rockfish_core::LibrarySpec;

/// The MIOpen library. Newest first.
pub const MIOPEN: LibrarySpec = LibrarySpec {
    logical_name: "MIOPEN",
    windows: &["MIOpen.dll", "libMIOpen.dll"],
    linux: &["libMIOpen.so.1", "libMIOpen.so"],
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
