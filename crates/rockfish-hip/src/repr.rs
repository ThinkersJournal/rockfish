//! Types that can live in device memory.

/// Marks a plain-data type whose bytes may be copied to and from the device.
///
/// # Safety
///
/// The type must be `Copy`, have no padding bytes, hold no pointers or
/// references, and accept every bit pattern as a valid value, because
/// [`crate::DeviceBuffer`] fills host memory with whatever the device holds.
pub unsafe trait DeviceRepr: Copy + 'static {}

// SAFETY: the primitive integers and floats satisfy every clause above.
unsafe impl DeviceRepr for u8 {}
unsafe impl DeviceRepr for i8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for f64 {}

// SAFETY: `half::f16` and `half::bf16` are `#[repr(transparent)]` over `u16`.
#[cfg(feature = "half")]
unsafe impl DeviceRepr for half::f16 {}
#[cfg(feature = "half")]
unsafe impl DeviceRepr for half::bf16 {}
