//! The error type of the safe layer.

use std::ffi::CStr;

use rockfish_core::LoadError;
use rockfish_hiprtc_sys::*;

/// Why a `rockfish-hiprtc` call failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The hipRTC library, or the entry point for this call, is absent.
    #[error(transparent)]
    Load(#[from] LoadError),
    /// A hipRTC call returned a non-success status.
    #[error("{call} failed: {name} ({code})")]
    Api {
        /// The hipRTC function that failed.
        call: &'static str,
        /// The raw `hiprtcResult`.
        code: i32,
        /// Its name, such as `HIPRTC_ERROR_INVALID_OPTION`.
        name: String,
    },
    /// The source did not compile; `log` is the compiler's output.
    #[error("compilation failed:\n{log}")]
    Compile {
        /// The program log at the time of failure.
        log: String,
    },
    /// The safe layer refused the call before reaching hipRTC.
    #[error("{0}")]
    Misuse(&'static str),
}

/// `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Name of a status code, from the constants in the vendored `hiprtc.h`.
#[allow(non_upper_case_globals)] // the bindgen constants are named `hiprtcResult_...`
pub fn builtin_name(code: i32) -> Option<&'static str> {
    Some(match code {
        hiprtcResult_HIPRTC_SUCCESS => "HIPRTC_SUCCESS",
        hiprtcResult_HIPRTC_ERROR_OUT_OF_MEMORY => "HIPRTC_ERROR_OUT_OF_MEMORY",
        hiprtcResult_HIPRTC_ERROR_PROGRAM_CREATION_FAILURE => {
            "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE"
        }
        hiprtcResult_HIPRTC_ERROR_INVALID_INPUT => "HIPRTC_ERROR_INVALID_INPUT",
        hiprtcResult_HIPRTC_ERROR_INVALID_PROGRAM => "HIPRTC_ERROR_INVALID_PROGRAM",
        hiprtcResult_HIPRTC_ERROR_INVALID_OPTION => "HIPRTC_ERROR_INVALID_OPTION",
        hiprtcResult_HIPRTC_ERROR_COMPILATION => "HIPRTC_ERROR_COMPILATION",
        hiprtcResult_HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE => {
            "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE"
        }
        hiprtcResult_HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION => {
            "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
        }
        hiprtcResult_HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION => {
            "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
        }
        hiprtcResult_HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID => {
            "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID"
        }
        hiprtcResult_HIPRTC_ERROR_INTERNAL_ERROR => "HIPRTC_ERROR_INTERNAL_ERROR",
        hiprtcResult_HIPRTC_ERROR_LINKING => "HIPRTC_ERROR_LINKING",
        _ => return None,
    })
}

/// Turn a hipRTC status into `Ok(())` or an [`Error::Api`] naming `call`.
///
/// The name comes from `hiprtcGetErrorString` when the library has it, else
/// from the built-in table, else `hiprtcResult(<code>)`.
pub(crate) fn check(api: &HiprtcApi, call: &'static str, code: i32) -> Result<()> {
    if code == hiprtcResult_HIPRTC_SUCCESS {
        return Ok(());
    }
    let from_library = api.hiprtcGetErrorString().ok().and_then(|f| {
        // SAFETY: `f` is hipRTC's `hiprtcGetErrorString`, which accepts any
        // status and returns a NUL-terminated static string or null.
        let p = unsafe { f(code as _) };
        // SAFETY: a non-null result is a valid NUL-terminated string.
        (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    });
    let name = from_library
        .or_else(|| builtin_name(code).map(str::to_owned))
        .unwrap_or_else(|| format!("hiprtcResult({code})"));
    Err(Error::Api { call, code, name })
}
