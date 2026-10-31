//! Programs: source in, device code object out.

use std::ffi::{CStr, CString, c_char, c_int};

use rockfish_hiprtc_sys::{hiprtcProgram, hiprtcResult_HIPRTC_ERROR_COMPILATION};

use crate::error::{Error, Result, check};
use crate::hiprtc;

/// The hipRTC library version, as reported by `hiprtcVersion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    /// Major version.
    pub major: i32,
    /// Minor version.
    pub minor: i32,
}

/// The loaded hipRTC library version.
pub fn version() -> Result<Version> {
    let api = hiprtc()?;
    let f = api.hiprtcVersion()?;
    let (mut major, mut minor) = (0, 0);
    // SAFETY: both out-pointers are valid for the call.
    check(api, "hiprtcVersion", unsafe { f(&mut major, &mut minor) })?;
    Ok(Version { major, minor })
}

/// A HIP C++ source being compiled for the GPU.
#[derive(Debug)]
pub struct Program {
    raw: hiprtcProgram,
}

// SAFETY: the handle is not tied to a thread. The rules for concurrent use
// of one program are unverified, so `Program` is `Send` but not `Sync`.
unsafe impl Send for Program {}

/// `s` as a C string, or [`Error::Misuse`] naming `what` if it holds a NUL.
fn c_string(s: &str, what: &'static str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::Misuse(what))
}

/// A hipRTC log buffer: the reported size counts a trailing NUL, which is dropped.
fn log_from_buffer(mut buf: Vec<u8>) -> String {
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}

impl Program {
    /// Create a program from `source`, named `name` in diagnostics (a file
    /// name such as `kernels.hip`).
    pub fn new(source: &str, name: &str) -> Result<Self> {
        let source = c_string(source, "source contains a NUL byte")?;
        let name = c_string(name, "program name contains a NUL byte")?;
        let api = hiprtc()?;
        let f = api.hiprtcCreateProgram()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer, both strings are
        // NUL-terminated, and zero headers means the two header arrays are
        // never read.
        check(api, "hiprtcCreateProgram", unsafe {
            f(
                &mut raw,
                source.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        })?;
        Ok(Self { raw })
    }

    /// Register a name expression (such as `add_one<float>` for a template)
    /// whose lowered, mangled name [`Program::lowered_name`] returns after
    /// compilation. Must be called before [`Program::compile`].
    pub fn add_name_expression(&mut self, expression: &str) -> Result<()> {
        let expression = c_string(expression, "name expression contains a NUL byte")?;
        let api = hiprtc()?;
        let f = api.hiprtcAddNameExpression()?;
        // SAFETY: `self.raw` is a live program and the string NUL-terminated.
        check(api, "hiprtcAddNameExpression", unsafe {
            f(self.raw, expression.as_ptr())
        })
    }

    /// Compile with `options` (compiler flags such as `--offload-arch=gfx1036`).
    ///
    /// A source error is [`Error::Compile`] carrying the log; warnings stay
    /// available from [`Program::log`] after a success.
    pub fn compile(&mut self, options: &[&str]) -> Result<()> {
        let owned = options
            .iter()
            .map(|o| c_string(o, "compile option contains a NUL byte"))
            .collect::<Result<Vec<_>>>()?;
        let pointers: Vec<*const c_char> = owned.iter().map(|o| o.as_ptr()).collect();
        let count = c_int::try_from(pointers.len())
            .map_err(|_| Error::Misuse("too many compile options"))?;
        let api = hiprtc()?;
        let f = api.hiprtcCompileProgram()?;
        // SAFETY: `self.raw` is live; `pointers` holds `count` pointers into
        // `owned`, which outlives the call.
        let code = unsafe { f(self.raw, count, pointers.as_ptr()) };
        if code == hiprtcResult_HIPRTC_ERROR_COMPILATION {
            return Err(Error::Compile { log: self.log()? });
        }
        check(api, "hiprtcCompileProgram", code)
    }

    /// The compiler log (diagnostics and warnings); empty if there is none.
    pub fn log(&self) -> Result<String> {
        let api = hiprtc()?;
        let size_fn = api.hiprtcGetProgramLogSize()?;
        let log_fn = api.hiprtcGetProgramLog()?;
        let mut size = 0usize;
        // SAFETY: `self.raw` is live and `size` a valid out-pointer.
        check(api, "hiprtcGetProgramLogSize", unsafe {
            size_fn(self.raw, &mut size)
        })?;
        if size == 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` has the `size` bytes hipRTC reported.
        check(api, "hiprtcGetProgramLog", unsafe {
            log_fn(self.raw, buf.as_mut_ptr().cast())
        })?;
        Ok(log_from_buffer(buf))
    }

    /// The compiled device code object, after a successful [`Program::compile`].
    pub fn code(&self) -> Result<Vec<u8>> {
        let api = hiprtc()?;
        let size_fn = api.hiprtcGetCodeSize()?;
        let code_fn = api.hiprtcGetCode()?;
        let mut size = 0usize;
        // SAFETY: `self.raw` is live and `size` a valid out-pointer.
        check(api, "hiprtcGetCodeSize", unsafe {
            size_fn(self.raw, &mut size)
        })?;
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` has the `size` bytes hipRTC reported.
        check(api, "hiprtcGetCode", unsafe {
            code_fn(self.raw, buf.as_mut_ptr().cast())
        })?;
        Ok(buf)
    }

    /// The mangled name of a registered name expression, for
    /// `Module::function`. Copied out: hipRTC owns the original.
    pub fn lowered_name(&self, expression: &str) -> Result<CString> {
        let expression = c_string(expression, "name expression contains a NUL byte")?;
        let api = hiprtc()?;
        let f = api.hiprtcGetLoweredName()?;
        let mut lowered: *const c_char = std::ptr::null();
        // SAFETY: `self.raw` is live, the string NUL-terminated and `lowered`
        // a valid out-pointer.
        check(api, "hiprtcGetLoweredName", unsafe {
            f(self.raw, expression.as_ptr(), &mut lowered)
        })?;
        if lowered.is_null() {
            return Err(Error::Misuse("hipRTC returned no lowered name"));
        }
        // SAFETY: non-null, NUL-terminated, valid until the program is
        // destroyed; copied before returning.
        Ok(unsafe { CStr::from_ptr(lowered) }.to_owned())
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if let Ok(f) = hiprtc().and_then(|api| Ok(api.hiprtcDestroyProgram()?)) {
            // SAFETY: `self.raw` is a live program, destroyed exactly once here.
            let _ = unsafe { f(&mut self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nul_bytes_are_refused_before_any_call() {
        assert!(matches!(c_string("a\0b", "x"), Err(Error::Misuse("x"))));
        assert!(matches!(
            Program::new("int x;\0", "k.hip"),
            Err(Error::Misuse(_))
        ));
        assert!(c_string("--offload-arch=gfx1036", "x").is_ok());
    }

    #[test]
    fn log_drops_the_trailing_nul_and_only_that() {
        assert_eq!(log_from_buffer(b"warning: x\n\0".to_vec()), "warning: x\n");
        assert_eq!(log_from_buffer(vec![0]), "");
        assert_eq!(log_from_buffer(Vec::new()), "");
    }
}
