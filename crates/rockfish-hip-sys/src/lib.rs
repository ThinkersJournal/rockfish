// SPDX-License-Identifier: MIT OR Apache-2.0
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
    rustdoc::all,
    reason = "bindgen output is not ours to restyle; the generated bindings trip these lints by construction (regenerate with `gen-bindings`)"
)]
#[rustfmt::skip]
mod generated;
pub use generated::*;
