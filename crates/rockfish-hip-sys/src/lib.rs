//! Raw, runtime-loaded bindings to the AMD HIP runtime.
//!
//! Nothing links against ROCm at build time. [`HipApi::load`] searches for the
//! HIP runtime library and resolves its entry points; asking for one the
//! loaded library lacks returns an error for that call only.
//!
//! The types and the table in `generated` are produced by `tools/gen-bindings`
//! from the HIP headers in `vendor/hip` (see `THIRD-PARTY-NOTICES.md`). Do not
//! edit them by hand.
//!
//! The library names below are the part that needs a real install to confirm:
//! `amdhip64_6.dll` was measured on a Windows laptop (driver-bundled runtime,
//! HIP 6.4.50501); the other names are unverified.

use rockfish_core::LibrarySpec;

/// The HIP runtime library. Newest first.
pub const HIP: LibrarySpec = LibrarySpec {
    logical_name: "HIP",
    windows: &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"],
    linux: &["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"],
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
