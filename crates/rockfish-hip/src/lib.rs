//! Safe Rust layer over the AMD HIP runtime.
//!
//! Nothing links against ROCm at build time. The runtime is loaded on first
//! use by [`hip`]; every call returns an [`Error`] rather than panicking, and
//! a runtime that is missing or lacks an entry point is an [`Error::Load`].
//!
//! Like Baracuda's `driver()`, the loaded API is one process-wide table and
//! handles do not hold it. Only the Windows driver runtime `amdhip64_6.dll`
//! (Radeon 610M, HIP 6.4) has been measured; other platforms are unverified.

mod buffer;
mod device;
mod error;
mod repr;
mod stream;
#[cfg(all(test, any(windows, target_os = "linux")))]
mod test_support;

use std::sync::{Mutex, OnceLock, PoisonError};

use rockfish_hip_sys::HipApi;

pub use buffer::DeviceBuffer;
pub use device::{Device, MemoryInfo, device_count, runtime_version};
pub use error::{Error, Result, builtin_name};
pub use repr::DeviceRepr;
pub use rockfish_core::LoadError;
pub use stream::{Event, Stream};

/// The process-wide HIP table, loading the runtime and warming it up on first use.
///
/// Warm-up (see [`warm_up`]) runs once, under a lock, before any thread gets
/// the table. A failed load is not cached, so installing HIP and calling
/// again works.
pub fn hip() -> Result<&'static HipApi> {
    static HIP: OnceLock<HipApi> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(api) = HIP.get() {
        return Ok(api);
    }
    let _guard = INIT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(api) = HIP.get() {
        return Ok(api);
    }
    let api = HipApi::load()?;
    warm_up(&api);
    Ok(HIP.get_or_init(|| api))
}

/// Create and destroy one stream on device 0, ignoring every failure.
///
/// Measured on the Windows laptop (`amdhip64_6.dll`, Radeon 610M): when one
/// thread's first `hipStreamCreate` overlapped other threads' first HIP calls,
/// the process hung (every run of the parallel test suite); creating a stream
/// first, serially, removed the hang. Why is not known. Other runtimes and
/// multi-device machines are unmeasured, so this covers device 0 only.
fn warm_up(api: &HipApi) {
    let (Ok(count), Ok(set), Ok(create), Ok(destroy)) = (
        api.hipGetDeviceCount(),
        api.hipSetDevice(),
        api.hipStreamCreate(),
        api.hipStreamDestroy(),
    ) else {
        return;
    };
    let mut n = 0;
    // SAFETY: `n` is a valid out-pointer.
    if unsafe { count(&mut n) } != 0 || n < 1 {
        return;
    }
    let mut stream = std::ptr::null_mut();
    // SAFETY: plain arguments; the stream is destroyed once, only if created.
    unsafe {
        if set(0) == 0 && create(&mut stream) == 0 {
            destroy(stream);
        }
    }
}
