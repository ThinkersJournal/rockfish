//! Pinned (page-locked) host memory and the guard for asynchronous copies.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::buffer::byte_len;
use crate::device::Device;
use crate::error::{Result, check};
use crate::hip;
use crate::repr::DeviceRepr;
use crate::stream::Stream;

/// Page-locked host memory holding `len` values of `T`, freed on drop.
///
/// Asynchronous copies need it: the driver can only overlap a transfer with
/// other work when the host side cannot be paged out. Contents start
/// unspecified (any bit pattern is a valid `T: DeviceRepr`). It dereferences
/// to `[T]`.
#[derive(Debug)]
pub struct PinnedBuffer<T: DeviceRepr> {
    ptr: *mut c_void,
    len: usize,
    _element: PhantomData<T>,
}

// SAFETY: the buffer owns its allocation and `T` is plain data.
unsafe impl<T: DeviceRepr> Send for PinnedBuffer<T> {}
// SAFETY: shared access is `&[T]` only.
unsafe impl<T: DeviceRepr> Sync for PinnedBuffer<T> {}

impl<T: DeviceRepr> PinnedBuffer<T> {
    /// Allocate room for `len` values, made current on `device` first.
    pub fn alloc(device: Device, len: usize) -> Result<Self> {
        let bytes = byte_len::<T>(len)?;
        device.set_current()?;
        let api = hip()?;
        let f = api.hipHostMalloc()?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer; flags 0 is `hipHostMallocDefault`.
        check(api, "hipHostMalloc", unsafe { f(&mut ptr, bytes, 0) })?;
        Ok(Self {
            ptr,
            len,
            _element: PhantomData,
        })
    }

    /// Allocate and fill from `data`.
    pub fn from_slice(device: Device, data: &[T]) -> Result<Self> {
        let mut buffer = Self::alloc(device, data.len())?;
        buffer.copy_from_slice(data);
        Ok(buffer)
    }
}

impl<T: DeviceRepr> Deref for PinnedBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` holds `len` values, any bit pattern valid, for as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
    }
}

impl<T: DeviceRepr> DerefMut for PinnedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as `deref`, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len) }
    }
}

impl<T: DeviceRepr> PinnedBuffer<T> {
    pub(crate) fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
    pub(crate) fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }
}

impl<T: DeviceRepr> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        // A failure to free cannot be reported from `drop`.
        if let Ok(f) = hip().and_then(|api| Ok(api.hipHostFree()?)) {
            // SAFETY: `self.ptr` came from `hipHostMalloc` and is freed once here.
            let _ = unsafe { f(self.ptr) };
        }
    }
}

/// An asynchronous copy that has been queued on a stream.
///
/// It borrows the host buffer, the device buffer and the stream, so none of
/// them can be freed, moved or touched while the copy may still be running.
/// [`PendingCopy::wait`] synchronises the stream and releases the borrows;
/// dropping the guard does the same, ignoring any error.
///
/// Caveat: `std::mem::forget` on the guard ends the borrows without waiting,
/// the same hole every scope-guard API in Rust has. Do not forget it.
#[must_use = "dropping the guard blocks until the copy finishes; call `wait` to see errors"]
#[derive(Debug)]
pub struct PendingCopy<'a> {
    stream: &'a Stream,
    done: bool,
}

impl<'a> PendingCopy<'a> {
    pub(crate) fn new(stream: &'a Stream) -> Self {
        Self {
            stream,
            done: false,
        }
    }

    /// Block until everything queued on the stream, this copy included, has finished.
    pub fn wait(mut self) -> Result<()> {
        self.done = true;
        self.stream.synchronize()
    }
}

impl Drop for PendingCopy<'_> {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.stream.synchronize();
        }
    }
}
