//! The error type of the safe layer.

use std::ffi::CStr;

use rockfish_core::LoadError;
use rockfish_rocblas_sys::*;

/// Why a `rockfish-rocblas` call failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The rocBLAS library, or the entry point for this call, is absent.
    #[error(transparent)]
    Load(#[from] LoadError),
    /// A `rockfish-hip` call made on rocBLAS's behalf failed.
    #[error(transparent)]
    Hip(#[from] rockfish_hip::Error),
    /// A rocBLAS call returned a non-success status.
    #[error("{call} failed: {name} ({code})")]
    Api {
        /// The rocBLAS function that failed.
        call: &'static str,
        /// The raw `rocblas_status`.
        code: i32,
        /// Its name, such as `rocblas_status_invalid_size`.
        name: String,
    },
    /// The safe layer refused the call before reaching rocBLAS.
    #[error("{0}")]
    Misuse(&'static str),
}

/// `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Name of a status code, from the constants in the vendored `rocblas-types.h`.
#[allow(non_upper_case_globals)] // the bindgen constants are named `rocblas_status__...`
pub fn builtin_name(code: i32) -> Option<&'static str> {
    Some(match code {
        rocblas_status__rocblas_status_success => "rocblas_status_success",
        rocblas_status__rocblas_status_invalid_handle => "rocblas_status_invalid_handle",
        rocblas_status__rocblas_status_not_implemented => "rocblas_status_not_implemented",
        rocblas_status__rocblas_status_invalid_pointer => "rocblas_status_invalid_pointer",
        rocblas_status__rocblas_status_invalid_size => "rocblas_status_invalid_size",
        rocblas_status__rocblas_status_memory_error => "rocblas_status_memory_error",
        rocblas_status__rocblas_status_internal_error => "rocblas_status_internal_error",
        rocblas_status__rocblas_status_perf_degraded => "rocblas_status_perf_degraded",
        rocblas_status__rocblas_status_size_query_mismatch => "rocblas_status_size_query_mismatch",
        rocblas_status__rocblas_status_size_increased => "rocblas_status_size_increased",
        rocblas_status__rocblas_status_size_unchanged => "rocblas_status_size_unchanged",
        rocblas_status__rocblas_status_invalid_value => "rocblas_status_invalid_value",
        rocblas_status__rocblas_status_continue => "rocblas_status_continue",
        rocblas_status__rocblas_status_check_numerics_fail => "rocblas_status_check_numerics_fail",
        rocblas_status__rocblas_status_excluded_from_build => "rocblas_status_excluded_from_build",
        rocblas_status__rocblas_status_arch_mismatch => "rocblas_status_arch_mismatch",
        _ => return None,
    })
}

/// Turn a rocBLAS status into `Ok(())` or an [`Error::Api`] naming `call`.
///
/// The name comes from `rocblas_status_to_string` when the library has it,
/// else from the built-in table, else `rocblas_status(<code>)`.
pub(crate) fn check(api: &RocblasApi, call: &'static str, code: i32) -> Result<()> {
    if code == rocblas_status__rocblas_status_success {
        return Ok(());
    }
    let from_library = api.rocblas_status_to_string().ok().and_then(|f| {
        // SAFETY: `f` is rocBLAS's `rocblas_status_to_string`, which accepts any
        // status and returns a NUL-terminated static string or null.
        let p = unsafe { f(code as _) };
        // SAFETY: a non-null result is a valid NUL-terminated string.
        (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    });
    let name = from_library
        .or_else(|| builtin_name(code).map(str::to_owned))
        .unwrap_or_else(|| format!("rocblas_status({code})"));
    Err(Error::Api { call, code, name })
}
