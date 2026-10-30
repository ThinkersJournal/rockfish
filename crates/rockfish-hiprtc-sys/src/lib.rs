//! Raw, runtime-loaded bindings to the AMD hipRTC runtime compiler.
//!
//! Like `rockfish-hip-sys`, nothing links against ROCm at build time, and a
//! missing library or entry point is an error value. The bindings in
//! `generated` come from `tools/gen-bindings` and the vendored `hiprtc.h`.
//!
//! hipRTC may be absent: the HIP runtime that ships with a Windows graphics
//! driver was measured on one laptop with no hipRTC beside it. The names below
//! are unverified guesses from the Linux soname convention and need a real
//! install to confirm.

use rockfish_core::LibrarySpec;

/// The hipRTC library. Newest first.
pub const HIPRTC: LibrarySpec = LibrarySpec {
    logical_name: "HIPRTC",
    windows: &["hiprtc.dll"],
    linux: &["libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so"],
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
