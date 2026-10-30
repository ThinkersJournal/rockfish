//! Device memory.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;

use rockfish_hip_sys::{
    hipMemcpyKind_hipMemcpyDeviceToDevice, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice,
};

use crate::device::Device;
use crate::error::{Error, Result, check};
use crate::hip;
use crate::repr::DeviceRepr;

/// `len` elements of `T` as a byte count; zero-sized requests and overflow are refused.
fn byte_len<T: DeviceRepr>(len: usize) -> Result<usize> {
    match len.checked_mul(size_of::<T>()) {
        Some(0) => Err(Error::Misuse("zero-length device allocation")),
        Some(bytes) => Ok(bytes),
        None => Err(Error::Misuse("device allocation size overflows usize")),
    }
}

fn same_len(device_len: usize, host_len: usize) -> Result<()> {
    if device_len == host_len {
        Ok(())
    } else {
        Err(Error::Misuse("buffer and slice lengths differ"))
    }
}

/// A block of device memory holding `len` values of `T`, freed on drop.
#[derive(Debug)]
pub struct DeviceBuffer<T: DeviceRepr> {
    ptr: *mut c_void,
    len: usize,
    device: Device,
    _element: PhantomData<T>,
}

// SAFETY: the buffer owns its allocation, `T` is plain data, and every call
// makes the buffer's device current on the calling thread first.
unsafe impl<T: DeviceRepr> Send for DeviceBuffer<T> {}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Allocate room for `len` values on `device`, contents unspecified.
    pub fn alloc(device: Device, len: usize) -> Result<Self> {
        let bytes = byte_len::<T>(len)?;
        device.set_current()?;
        let api = hip()?;
        let f = api.hipMalloc()?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer for the call.
        check(api, "hipMalloc", unsafe { f(&mut ptr, bytes) })?;
        Ok(Self {
            ptr,
            len,
            device,
            _element: PhantomData,
        })
    }

    /// Allocate a buffer and copy `data` into it.
    pub fn from_slice(device: Device, data: &[T]) -> Result<Self> {
        let mut buffer = Self::alloc(device, data.len())?;
        buffer.copy_from_host(data)?;
        Ok(buffer)
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: empty buffers cannot be created.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device that owns the memory.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The device address, for kernel arguments. Valid until the buffer is dropped.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Copy `data` from the host into this buffer. Blocks until done.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        same_len(self.len, data.len())?;
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipMemcpy()?;
        let bytes = std::mem::size_of_val(data);
        // SAFETY: `self.ptr` holds `self.len` values and `data` is that long, as checked.
        check(api, "hipMemcpy", unsafe {
            f(
                self.ptr,
                data.as_ptr().cast(),
                bytes,
                hipMemcpyKind_hipMemcpyHostToDevice,
            )
        })
    }

    /// Copy this buffer into `out` on the host. Blocks until done.
    pub fn copy_to_host(&self, out: &mut [T]) -> Result<()> {
        same_len(self.len, out.len())?;
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipMemcpy()?;
        let bytes = std::mem::size_of_val(out);
        // SAFETY: `self.ptr` holds `self.len` values and `out` is that long, as checked.
        check(api, "hipMemcpy", unsafe {
            f(
                out.as_mut_ptr().cast(),
                self.ptr,
                bytes,
                hipMemcpyKind_hipMemcpyDeviceToHost,
            )
        })
    }

    /// The contents as a new `Vec`. Blocks until done.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut out: Vec<T> = Vec::with_capacity(self.len);
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipMemcpy()?;
        // SAFETY: `out` has capacity for `self.len` values, the device fills
        // them all, and `T: DeviceRepr` accepts any bit pattern.
        unsafe {
            check(api, "hipMemcpy", {
                f(
                    out.as_mut_ptr().cast(),
                    self.ptr,
                    self.len * size_of::<T>(),
                    hipMemcpyKind_hipMemcpyDeviceToHost,
                )
            })?;
            out.set_len(self.len);
        }
        Ok(out)
    }

    /// Copy another buffer on the same device into this one.
    pub fn copy_from_device(&mut self, source: &DeviceBuffer<T>) -> Result<()> {
        same_len(self.len, source.len)?;
        if self.device != source.device {
            return Err(Error::Misuse(
                "copies between devices are not supported yet",
            ));
        }
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipMemcpy()?;
        // SAFETY: both allocations hold `self.len` values, on the current device.
        check(api, "hipMemcpy", unsafe {
            f(
                self.ptr,
                source.ptr,
                self.len * size_of::<T>(),
                hipMemcpyKind_hipMemcpyDeviceToDevice,
            )
        })
    }

    /// Set every byte to zero.
    pub fn fill_zero(&mut self) -> Result<()> {
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipMemset()?;
        // SAFETY: `self.ptr` holds `self.len` values.
        check(api, "hipMemset", unsafe {
            f(self.ptr, 0, self.len * size_of::<T>())
        })
    }
}

impl<T: DeviceRepr> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // A failure to free cannot be reported from `drop`.
        if self.device.set_current().is_err() {
            return;
        }
        if let Ok(f) = hip().and_then(|api| Ok(api.hipFree()?)) {
            // SAFETY: `self.ptr` came from `hipMalloc` and is freed exactly once here.
            let _ = unsafe { f(self.ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_overflowing_sizes_are_refused_before_hip_is_asked() {
        assert!(matches!(byte_len::<u32>(0), Err(Error::Misuse(_))));
        assert!(matches!(byte_len::<u64>(usize::MAX), Err(Error::Misuse(_))));
        assert_eq!(byte_len::<u32>(3).unwrap(), 12);
    }

    #[cfg(feature = "half")]
    #[test]
    fn half_types_are_two_bytes_and_round_trip() {
        assert_eq!(byte_len::<half::f16>(3).unwrap(), 6);
        assert_eq!(byte_len::<half::bf16>(3).unwrap(), 6);
        let Some(device) = device() else { return };
        let sent: Vec<half::bf16> = (0..64)
            .map(|i| half::bf16::from_f32(i as f32 * 0.5))
            .collect();
        let buffer = DeviceBuffer::from_slice(device, &sent).expect("upload");
        assert_eq!(buffer.to_vec().expect("download"), sent);
    }

    #[test]
    fn length_mismatch_is_refused() {
        assert!(matches!(same_len(4, 5), Err(Error::Misuse(_))));
        assert!(same_len(4, 4).is_ok());
    }

    // GPU tests: UNVERIFIED in CI (no GPU on the GitHub runners); measured
    // locally on the Radeon 610M. Never `#[ignore]`: without HIP they say so
    // and stop.
    fn device() -> Option<Device> {
        match crate::device_count() {
            Ok(n) if n > 0 => Some(Device::get(0).expect("device 0")),
            _ => {
                eprintln!("no HIP device on this machine: GPU assertions not exercised");
                None
            }
        }
    }

    #[test]
    fn host_device_round_trip_returns_what_was_sent_not_the_prefill() {
        let Some(device) = device() else { return };
        let sent: Vec<u32> = (0..1000u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let buffer = DeviceBuffer::from_slice(device, &sent).expect("upload");
        // Prefill the destination with a different pattern so an unwritten
        // destination cannot pass.
        let mut back = vec![0xDEAD_BEEFu32; sent.len()];
        buffer.copy_to_host(&mut back).expect("download");
        assert_eq!(back, sent);
        assert_eq!(buffer.to_vec().expect("to_vec"), sent);
    }

    #[test]
    fn device_to_device_copy_and_fill_zero() {
        let Some(device) = device() else { return };
        let src = DeviceBuffer::from_slice(device, &[7u8; 256]).expect("src");
        let mut dst = DeviceBuffer::<u8>::alloc(device, 256).expect("dst");
        dst.copy_from_device(&src).expect("d2d");
        assert_eq!(dst.to_vec().unwrap(), vec![7u8; 256]);
        dst.fill_zero().expect("zero");
        assert_eq!(dst.to_vec().unwrap(), vec![0u8; 256]);
        let mut short = DeviceBuffer::<u8>::alloc(device, 128).expect("short");
        assert!(matches!(
            short.copy_from_device(&src),
            Err(Error::Misuse(_))
        ));
    }

    #[test]
    fn impossible_allocation_is_an_api_error_with_a_name() {
        let Some(device) = device() else { return };
        match DeviceBuffer::<u8>::alloc(device, usize::MAX / 2) {
            Err(Error::Api { call, name, .. }) => {
                assert_eq!(call, "hipMalloc");
                eprintln!("huge allocation reported as {name}");
                assert!(name.starts_with("hipError"), "name was {name}");
            }
            other => panic!("expected an Api error, got {:?}", other.map(|b| b.len())),
        }
    }

    #[test]
    fn events_record_complete_and_report_a_non_negative_interval() {
        use crate::{Event, Stream};
        let Some(device) = device() else { return };
        let stream = Stream::new(device).expect("stream");
        let (start, end) = (Event::new(device).unwrap(), Event::new(device).unwrap());
        start.record(&stream).unwrap();
        end.record(&stream).unwrap();
        end.synchronize().unwrap();
        stream.synchronize().unwrap();
        assert!(end.is_done().unwrap());
        assert!(stream.is_idle().unwrap());
        assert!(end.elapsed_since(&start).unwrap() >= 0.0);
    }
}
