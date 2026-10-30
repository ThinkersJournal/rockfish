//! Loader behaviour of the generated table. None of these need ROCm.

use rockfish_core::{Library, LibrarySpec, LoadError};
use rockfish_hip_sys::HipApi;

// A library every machine of that OS has, standing in for "a library that loads but is not HIP".
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
fn a_library_without_hip_symbols_reports_each_one_missing_not_panics() {
    let api = HipApi::from_library(Library::open(&OTHER).expect("system library loads"));
    match api.hipMalloc() {
        Err(LoadError::SymbolMissing { symbol, .. }) => assert_eq!(symbol, "hipMalloc"),
        other => panic!("expected SymbolMissing, got {:?}", other.map(|_| ())),
    }
    assert!(api.hipGetDeviceCount().is_err());
    assert!(api.hipModuleLaunchKernel().is_err());
}

#[test]
fn load_without_hip_is_an_error_and_with_hip_reports_a_device_count() {
    match HipApi::load() {
        Err(LoadError::LibraryNotFound { library, tried }) => {
            assert_eq!(library, "HIP");
            assert!(!tried.is_empty());
        }
        Err(LoadError::UnsupportedPlatform { .. }) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(api) => {
            // A HIP runtime is installed: ask it how many devices it has.
            let mut count = -1;
            let get_count = api
                .hipGetDeviceCount()
                .expect("runtime exports hipGetDeviceCount");
            // SAFETY: `count` is a valid out-pointer for the call.
            let status = unsafe { get_count(&mut count) };
            assert_eq!(status, 0, "hipGetDeviceCount failed on a machine with HIP");
            assert!(count >= 0);
        }
    }
}
