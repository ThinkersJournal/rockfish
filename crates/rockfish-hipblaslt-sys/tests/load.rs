//! Loader behaviour of the generated table. None of these need ROCm, except that
//! the last test checks every bound symbol when a hipBLASLt library can be found.

use rockfish_core::LoadError;
#[cfg(any(windows, target_os = "linux"))]
use rockfish_core::{Library, LibrarySpec};
use rockfish_hipblaslt_sys::HipblasltApi;

// The number of functions in `tools/gen-bindings` for hipBLASLt.
const BOUND: usize = 24;

// A library every machine of that OS has, standing in for "a library that loads but is not hipBLASLt".
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
fn a_library_without_hipblaslt_symbols_reports_each_one_missing_not_panics() {
    let api = HipblasltApi::from_library(Library::open(&OTHER).expect("system library loads"));
    match api.hipblasLtMatmul() {
        Err(LoadError::SymbolMissing { symbol, .. }) => assert_eq!(symbol, "hipblasLtMatmul"),
        other => panic!("expected SymbolMissing, got {:?}", other.map(|_| ())),
    }
    assert!(api.hipblasLtCreate().is_err());
    // Negative control for the check below: it can fail, and counts every bound function.
    assert_eq!(api.unresolved().len(), BOUND);
}

#[test]
fn load_without_hipblaslt_is_an_error_and_with_it_every_bound_symbol_resolves() {
    match HipblasltApi::load() {
        Err(LoadError::LibraryNotFound { library, tried }) => {
            assert_eq!(library, "HIPBLASLT");
            assert!(!tried.is_empty());
            eprintln!(
                "no hipBLASLt found: the symbol check did not run (put its directory on PATH; see ROADMAP R6-L)"
            );
        }
        Err(LoadError::UnsupportedPlatform { .. }) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(api) => {
            assert_eq!(api.unresolved(), Vec::<&str>::new());
        }
    }
}
