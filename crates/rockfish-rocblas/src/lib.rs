//! Safe Rust layer over AMD rocBLAS.
//!
//! A [`Handle`] binds rocBLAS to a `rockfish-hip` device and, optionally, a
//! [`rockfish_hip::Stream`]; [`Handle::gemm`] and [`Handle::gemm_strided_batched`]
//! multiply column-major matrices held in [`rockfish_hip::DeviceBuffer`]s, for
//! `f32`, `f64` and (feature `half`) `half::f16` / `half::bf16`. Nothing links
//! against ROCm at build time, and rocBLAS is optional: a machine without it
//! gets [`Error::Load`].
//!
//! Every shape, leading dimension, buffer length and device mismatch is checked
//! before rocBLAS is called, and reported as [`Error::Misuse`].
//!
//! **UNVERIFIED on a GPU.** On the one machine measured (Windows laptop,
//! Radeon 610M gfx1036) the only rocBLAS is Ollama's ROCm 7.1 bundle, which has
//! no gfx1036 kernels and needs its own `amdhip64_7.dll`, so no GEMM has ever
//! run. Measured are the library load, the version string and the error paths;
//! the argument checks are unit-tested. GEMM correctness is a rig row.

mod error;
mod gemm;
mod handle;

use std::sync::{Mutex, OnceLock, PoisonError};

use rockfish_rocblas_sys::RocblasApi;

pub use error::{Error, Result, builtin_name};
pub use gemm::{GemmElement, Op};
pub use handle::Handle;
pub use rockfish_core::LoadError;

/// The process-wide rocBLAS table, loading the library on first use.
///
/// A failed load is not cached, so installing rocBLAS and calling again works.
pub fn rocblas() -> Result<&'static RocblasApi> {
    static API: OnceLock<RocblasApi> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let _guard = INIT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let api = RocblasApi::load()?;
    Ok(API.get_or_init(|| api))
}

/// The loaded rocBLAS version string, such as `5.1.1.<build>`.
pub fn version() -> Result<String> {
    version_with(rocblas()?)
}

fn version_with(api: &RocblasApi) -> Result<String> {
    let size = api.rocblas_get_version_string_size()?;
    let get = api.rocblas_get_version_string()?;
    let mut len = 0usize;
    // SAFETY: `len` is a valid out-pointer.
    error::check(api, "rocblas_get_version_string_size", unsafe {
        size(&mut len)
    })?;
    // Room for the terminator whether or not the reported size counts it.
    let mut buf = vec![0u8; len.max(64) + 1];
    // SAFETY: `buf` is writable for the stated length.
    error::check(api, "rocblas_get_version_string", unsafe {
        get(buf.as_mut_ptr().cast(), buf.len())
    })?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
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

    /// A loaded library that is not rocBLAS: every entry point is missing.
    pub(crate) fn not_rocblas() -> RocblasApi {
        RocblasApi::from_library(Library::open(&OTHER).expect("system library loads"))
    }

    #[test]
    fn a_missing_entry_point_is_an_error_value() {
        let api = not_rocblas();
        assert!(matches!(
            version_with(&api),
            Err(Error::Load(LoadError::SymbolMissing { .. }))
        ));
    }

    #[test]
    fn api_error_falls_back_to_the_builtin_name() {
        let api = not_rocblas();
        match error::check(&api, "rocblas_sgemm", 4) {
            Err(Error::Api { call, code, name }) => assert_eq!(
                (call, code, name.as_str()),
                ("rocblas_sgemm", 4, "rocblas_status_invalid_size")
            ),
            other => panic!("expected Api error, got {other:?}"),
        }
        match error::check(&api, "x", 424242) {
            Err(Error::Api { name, .. }) => assert_eq!(name, "rocblas_status(424242)"),
            other => panic!("expected Api error, got {other:?}"),
        }
        assert!(error::check(&api, "x", 0).is_ok());
    }

    #[test]
    fn builtin_names_cover_every_header_code() {
        for code in 0..=15 {
            assert!(builtin_name(code).is_some(), "no name for {code}");
        }
        assert_eq!(builtin_name(16), None);
    }

    /// Passes both ways: with rocBLAS absent the load error is the result;
    /// with it present (on `PATH`, or `ROCKFISH_ROCBLAS_PATH`) a version is.
    /// Which one ran is printed.
    #[test]
    fn absent_rocblas_is_an_error_not_a_panic() {
        match version() {
            Err(Error::Load(e)) => println!("rocBLAS absent, as expected here: {e}"),
            Ok(v) => {
                assert!(!v.is_empty());
                println!("rocBLAS present: {v}");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
