//! The rocBLAS handle.

use std::marker::PhantomData;

use rockfish_hip::{Device, Stream};
use rockfish_hip_sys::hipStream_t;
use rockfish_rocblas_sys::{rocblas_handle, rocblas_pointer_mode__rocblas_pointer_mode_host};

use crate::error::{Result, check};
use crate::rocblas;

/// A rocBLAS context bound to one device and, optionally, one stream.
///
/// Scalars (`alpha`, `beta`) are always host values. Work is queued on the
/// handle's stream (the default stream for [`Handle::new`]) and returns before
/// it finishes: call [`Handle::synchronize`] before reading results back on
/// the host. The lifetime `'s` ties a handle made by [`Handle::on_stream`] to
/// its stream, so the stream cannot be destroyed under it.
#[derive(Debug)]
pub struct Handle<'s> {
    pub(crate) raw: rocblas_handle,
    pub(crate) device: Device,
    stream: hipStream_t,
    _stream: PhantomData<&'s Stream>,
}

// SAFETY: a rocBLAS handle may be moved between threads as long as one thread
// uses it at a time (rocBLAS documents one handle per thread); each call makes
// the handle's device current on the calling thread first. `Handle` is `Send`
// but not `Sync`.
unsafe impl Send for Handle<'_> {}

impl Handle<'static> {
    /// A handle on `device`'s default stream.
    pub fn new(device: Device) -> Result<Self> {
        Self::create(device, std::ptr::null_mut())
    }
}

impl<'s> Handle<'s> {
    /// A handle whose work is queued on `stream`.
    pub fn on_stream(stream: &'s Stream) -> Result<Self> {
        Self::create(stream.device(), stream.as_raw())
    }

    fn create(device: Device, stream: hipStream_t) -> Result<Self> {
        device.set_current()?;
        let api = rocblas()?;
        let create = api.rocblas_create_handle()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer for the call.
        check(api, "rocblas_create_handle", unsafe { create(&mut raw) })?;
        // From here the handle is destroyed by `Drop` if a later step fails.
        let handle = Self {
            raw,
            device,
            stream,
            _stream: PhantomData,
        };
        let set_mode = api.rocblas_set_pointer_mode()?;
        // SAFETY: `raw` is a live handle.
        check(api, "rocblas_set_pointer_mode", unsafe {
            set_mode(raw, rocblas_pointer_mode__rocblas_pointer_mode_host)
        })?;
        if !stream.is_null() {
            let set_stream = api.rocblas_set_stream()?;
            // SAFETY: `raw` is a live handle and `stream` a live stream that
            // outlives it (`'s`).
            check(api, "rocblas_set_stream", unsafe {
                set_stream(raw, stream)
            })?;
        }
        Ok(handle)
    }

    /// The device this handle works on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Wait until everything queued on this handle's stream has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.device.set_current()?;
        let api = rockfish_hip::hip()?;
        let f = api.hipStreamSynchronize()?;
        // SAFETY: `self.stream` is null (the default stream) or a live stream.
        let code = unsafe { f(self.stream) };
        if code == 0 {
            Ok(())
        } else {
            Err(rockfish_hip::Error::Api {
                call: "hipStreamSynchronize",
                code,
                name: rockfish_hip::builtin_name(code)
                    .map_or_else(|| format!("hipError({code})"), str::to_owned),
            }
            .into())
        }
    }
}

impl Drop for Handle<'_> {
    fn drop(&mut self) {
        // A failure to destroy cannot be reported from `drop`.
        if self.device.set_current().is_err() {
            return;
        }
        if let Ok(f) = rocblas().and_then(|api| Ok(api.rocblas_destroy_handle()?)) {
            // SAFETY: `self.raw` is a live handle, destroyed exactly once here.
            let _ = unsafe { f(self.raw) };
        }
    }
}
