//! A loaded library that is not HIP, standing in for "a runtime missing every entry point".

use rockfish_core::{Library, LibrarySpec};
use rockfish_hip_sys::HipApi;

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
pub(crate) fn not_hip() -> HipApi {
    HipApi::from_library(Library::open(&OTHER).expect("system library loads"))
}
