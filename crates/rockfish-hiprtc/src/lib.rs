//! Safe Rust layer over the AMD hipRTC runtime compiler.
//!
//! Compiles HIP C++ source to a device code object at run time, ready for
//! `rockfish_hip::Module::from_code_object`. Nothing links against ROCm at
//! build time, and hipRTC is optional: a machine without it gets
//! [`Error::Load`] from [`hiprtc`] and every [`Program`] call.
//!
//! **UNVERIFIED end to end.** The only machine measured has no hipRTC
//! (`hiprtc*.dll` is absent beside the Windows driver's `amdhip64_6.dll`), so
//! the library names, the option spellings and the compile path have never run
//! against a real hipRTC. What is tested here is the absent-library path, the
//! missing-entry-point path, and the argument handling before any call.

mod error;
mod program;

use std::sync::{Mutex, OnceLock, PoisonError};

use rockfish_hiprtc_sys::HiprtcApi;

pub use error::{Error, Result, builtin_name};
pub use program::{Program, Version, version};
pub use rockfish_core::LoadError;

/// The process-wide hipRTC table, loading the library on first use.
///
/// A failed load is not cached, so installing hipRTC and calling again works.
pub fn hiprtc() -> Result<&'static HiprtcApi> {
    static API: OnceLock<HiprtcApi> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let _guard = INIT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let api = HiprtcApi::load()?;
    Ok(API.get_or_init(|| api))
}

#[cfg(all(test, any(windows, target_os = "linux")))]
mod tests {
    use rockfish_core::{Library, LibrarySpec};

    use super::*;

    #[cfg(windows)]
    const OTHER: LibrarySpec = LibrarySpec {
        logical_name: "Other",
        windows: &["kernel32.dll"],
        linux: &[],
    };
    #[cfg(target_os = "linux")]
    const OTHER: LibrarySpec = LibrarySpec {
        logical_name: "Other",
        windows: &[],
        linux: &["libc.so.6"],
    };

    /// A loaded library that is not hipRTC: every entry point is missing.
    fn not_hiprtc() -> HiprtcApi {
        HiprtcApi::from_library(Library::open(&OTHER).expect("system library loads"))
    }

    #[test]
    fn a_missing_entry_point_is_an_error_value() {
        let api = not_hiprtc();
        assert!(matches!(
            api.hiprtcCreateProgram(),
            Err(LoadError::SymbolMissing { .. })
        ));
    }

    #[test]
    fn api_error_falls_back_to_the_builtin_name() {
        let api = not_hiprtc();
        match error::check(&api, "hiprtcCompileProgram", 5) {
            Err(Error::Api { call, code, name }) => assert_eq!(
                (call, code, name.as_str()),
                ("hiprtcCompileProgram", 5, "HIPRTC_ERROR_INVALID_OPTION")
            ),
            other => panic!("expected Api error, got {other:?}"),
        }
        match error::check(&api, "x", 424242) {
            Err(Error::Api { name, .. }) => assert_eq!(name, "hiprtcResult(424242)"),
            other => panic!("expected Api error, got {other:?}"),
        }
        assert!(error::check(&api, "x", 0).is_ok());
    }

    #[test]
    fn builtin_names_cover_every_header_code() {
        for code in (0..=11).chain([100]) {
            assert!(builtin_name(code).is_some(), "no name for {code}");
        }
        assert_eq!(builtin_name(12), None);
    }

    /// Passes both ways: with hipRTC absent the load error is the result
    /// (measured on the Windows laptop); with it present, a version is. Which
    /// one ran is printed.
    #[test]
    fn absent_hiprtc_is_an_error_not_a_panic() {
        match version() {
            Err(Error::Load(e)) => println!("hipRTC absent, as expected here: {e}"),
            Ok(v) => println!("hipRTC present: {v:?}"),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
