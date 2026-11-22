//! Loader behaviour of the generated table. None of these need ROCm, except that
//! the last test checks every bound symbol when a rocSOLVER library can be found.

use rockfish_core::LoadError;
#[cfg(any(windows, target_os = "linux"))]
use rockfish_core::{Library, LibrarySpec};
use rockfish_rocsolver_sys::RocsolverApi;

// The number of functions in `tools/gen-bindings` for rocSOLVER.
const BOUND: usize = 36;

// A library every machine of that OS has, standing in for "a library that loads but is not rocSOLVER".
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

#[cfg(any(windows, target_os = "linux"))]
#[test]
fn a_library_without_rocsolver_symbols_reports_each_one_missing_not_panics() {
    let api = RocsolverApi::from_library(Library::open(&OTHER).expect("system library loads"));
    match api.rocsolver_dpotrf() {
        Err(LoadError::SymbolMissing { symbol, .. }) => {
            assert_eq!(symbol, "rocsolver_dpotrf")
        }
        other => panic!("expected SymbolMissing, got {:?}", other.map(|_| ())),
    }
    assert!(api.rocsolver_spotrf().is_err());
    // Negative control for the check below: it can fail, and counts every bound function.
    assert_eq!(api.unresolved().len(), BOUND);
}

#[test]
fn load_without_rocsolver_is_an_error_and_with_it_every_bound_symbol_resolves() {
    match RocsolverApi::load() {
        Err(LoadError::LibraryNotFound { library, tried }) => {
            assert_eq!(library, "ROCSOLVER");
            assert!(!tried.is_empty());
            eprintln!(
                "no rocSOLVER found: the symbol check did not run (no rocSOLVER was available when this was written)"
            );
        }
        Err(LoadError::UnsupportedPlatform { .. }) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(api) => {
            assert_eq!(api.unresolved(), Vec::<&str>::new());
        }
    }
}
