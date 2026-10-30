//! The error type of the safe layer.

use std::ffi::CStr;

use rockfish_core::LoadError;
use rockfish_hip_sys::HipApi;

use crate::error_names::NAMES;

/// Why a `rockfish-hip` call failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The HIP library, or the entry point for this call, is absent.
    #[error(transparent)]
    Load(#[from] LoadError),
    /// A HIP call returned a non-zero status.
    #[error("{call} failed: {name} ({code})")]
    Api {
        /// The HIP function that failed.
        call: &'static str,
        /// The raw `hipError_t`.
        code: i32,
        /// Its name, such as `hipErrorOutOfMemory`.
        name: String,
    },
    /// The safe layer refused the call before reaching HIP.
    #[error("{0}")]
    Misuse(&'static str),
}

/// `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Name of a status code from the built-in table (the names in the vendored
/// header), or `None` for a code the table does not know.
pub fn builtin_name(code: i32) -> Option<&'static str> {
    NAMES.iter().find(|(c, _)| *c == code).map(|(_, n)| *n)
}

/// Turn a HIP status into `Ok(())` or an [`Error::Api`] naming `call`.
///
/// The name comes from `hipGetErrorName` when the library has it, else from
/// the built-in table, else `hipError(<code>)`.
pub(crate) fn check(api: &HipApi, call: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let from_library = api.hipGetErrorName().ok().and_then(|f| {
        // SAFETY: `f` is HIP's `hipGetErrorName`, which accepts any status
        // value and returns a NUL-terminated static string or null.
        let p = unsafe { f(code as _) };
        // SAFETY: a non-null result is a valid NUL-terminated string.
        (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    });
    let name = from_library
        .or_else(|| builtin_name(code).map(str::to_owned))
        .unwrap_or_else(|| format!("hipError({code})"));
    Err(Error::Api { call, code, name })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::not_hip;

    #[test]
    fn table_has_a_name_for_every_status_constant_in_the_bindings() {
        let src = include_str!("../../rockfish-hip-sys/src/generated.rs");
        let mut seen = 0;
        for line in src.lines() {
            let Some(rest) = line.trim().strip_prefix("pub const hipError_t_") else {
                continue;
            };
            let (_, tail) = rest.split_once(": hipError_t = ").expect("constant shape");
            let code: i32 = tail.trim_end_matches(';').parse().expect("integer code");
            assert!(builtin_name(code).is_some(), "no name for code {code}");
            seen += 1;
        }
        // The control: the scan found the constants at all.
        assert!(seen > 50, "scan found only {seen} constants");
    }

    #[test]
    fn known_and_unknown_codes() {
        assert_eq!(builtin_name(0), Some("hipSuccess"));
        assert_eq!(builtin_name(2), Some("hipErrorOutOfMemory"));
        assert_eq!(builtin_name(-12345), None);
    }

    #[cfg(any(windows, target_os = "linux"))]
    #[test]
    fn api_error_falls_back_to_the_builtin_name_when_the_library_has_none() {
        let api = not_hip();
        match check(&api, "hipMalloc", 2) {
            Err(Error::Api { call, code, name }) => {
                assert_eq!(
                    (call, code, name.as_str()),
                    ("hipMalloc", 2, "hipErrorOutOfMemory")
                );
            }
            other => panic!("expected Api error, got {other:?}"),
        }
        match check(&api, "hipMalloc", 99999) {
            Err(Error::Api { name, .. }) => assert_eq!(name, "hipError(99999)"),
            other => panic!("expected Api error, got {other:?}"),
        }
        assert!(check(&api, "hipMalloc", 0).is_ok());
    }
}
