//! Safe Rust layer over the AMD HIP runtime.
//!
//! Nothing links against ROCm at build time. The runtime is loaded on first
//! use by [`hip`]; every call returns an [`Error`] rather than panicking, and
//! a runtime that is missing or lacks an entry point is an [`Error::Load`].
//!
//! Like Baracuda's `driver()`, the loaded API is one process-wide table and
//! handles do not hold it. Only the Windows driver runtime `amdhip64_6.dll`
//! (Radeon 610M, HIP 6.4) has been measured; other platforms are unverified.

mod device;
mod error;
#[cfg(all(test, any(windows, target_os = "linux")))]
mod test_support;

use std::sync::OnceLock;

use rockfish_hip_sys::HipApi;

pub use device::{Device, MemoryInfo, device_count, runtime_version};
pub use error::{Error, Result, builtin_name};
pub use rockfish_core::LoadError;

/// The process-wide HIP table, loading the runtime on first use.
///
/// A failed load is not cached, so installing HIP and calling again works.
pub fn hip() -> Result<&'static HipApi> {
    static HIP: OnceLock<HipApi> = OnceLock::new();
    if let Some(api) = HIP.get() {
        return Ok(api);
    }
    let api = HipApi::load()?;
    // If another thread won the race, ours is dropped and theirs is returned.
    Ok(HIP.get_or_init(|| api))
}
