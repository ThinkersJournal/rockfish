// SPDX-License-Identifier: MIT OR Apache-2.0
//! Loader behaviour of the generated table. None of these need ROCm, except that
//! the last test checks every bound symbol when a rocRAND library can be found.

use rockfish_core::LoadError;
#[cfg(any(windows, target_os = "linux"))]
use rockfish_core::{Library, LibrarySpec};
use rockfish_rocrand_sys::RocrandApi;

// The number of functions in `tools/gen-bindings` for rocRAND.
const BOUND: usize = 30;

// A library every machine of that OS has, standing in for "a library that loads but is not rocRAND".
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
fn a_library_without_rocrand_symbols_reports_each_one_missing_not_panics() {
    let api = RocrandApi::from_library(Library::open(&OTHER).expect("system library loads"));
    match api.rocrand_generate_uniform_double() {
        Err(LoadError::SymbolMissing { symbol, .. }) => {
            assert_eq!(symbol, "rocrand_generate_uniform_double")
        }
        other => panic!("expected SymbolMissing, got {:?}", other.map(|_| ())),
    }
    assert!(api.rocrand_generate_uniform().is_err());
    // Negative control for the check below: it can fail, and counts every bound function.
    assert_eq!(api.unresolved().len(), BOUND);
}

#[test]
fn load_without_rocrand_is_an_error_and_with_it_every_bound_symbol_resolves() {
    match RocrandApi::load() {
        Err(LoadError::LibraryNotFound { library, tried }) => {
            assert_eq!(library, "ROCRAND");
            assert!(!tried.is_empty());
            eprintln!(
                "no rocRAND found: the symbol check did not run (no rocRAND was available when this was written)"
            );
        }
        Err(LoadError::UnsupportedPlatform { .. }) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(api) => {
            assert_eq!(api.unresolved(), Vec::<&str>::new());
        }
    }
}
