//! Streams and events.

use rockfish_hip_sys::{hipEvent_t, hipStream_t};

use crate::device::Device;
use crate::error::{Result, check, check_ready};
use crate::hip;

/// An ordered queue of device work.
#[derive(Debug)]
pub struct Stream {
    raw: hipStream_t,
    device: Device,
}

// SAFETY: a stream handle may be used from any thread; each call makes the
// stream's device current on the calling thread first. HIP's rules for
// concurrent use of one stream from several threads are unverified, so
// `Stream` is `Send` but not `Sync`.
unsafe impl Send for Stream {}

impl Stream {
    /// Create a stream on `device`.
    pub fn new(device: Device) -> Result<Self> {
        device.set_current()?;
        let api = hip()?;
        let f = api.hipStreamCreate()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer for the call.
        check(api, "hipStreamCreate", unsafe { f(&mut raw) })?;
        Ok(Self { raw, device })
    }

    /// The device this stream belongs to.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The raw handle. Valid until this `Stream` is dropped.
    pub fn as_raw(&self) -> hipStream_t {
        self.raw
    }

    /// Wait until all work queued on the stream has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipStreamSynchronize()?;
        // SAFETY: `self.raw` is a live stream.
        check(api, "hipStreamSynchronize", unsafe { f(self.raw) })
    }

    /// Whether all queued work has finished, without waiting.
    pub fn is_idle(&self) -> Result<bool> {
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipStreamQuery()?;
        // SAFETY: `self.raw` is a live stream.
        check_ready(api, "hipStreamQuery", unsafe { f(self.raw) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // A failure to destroy cannot be reported from `drop`.
        if self.device.set_current().is_err() {
            return;
        }
        if let Ok(f) = hip().and_then(|api| Ok(api.hipStreamDestroy()?)) {
            // SAFETY: `self.raw` is a live stream, destroyed exactly once here.
            let _ = unsafe { f(self.raw) };
        }
    }
}

/// A point in a stream, for ordering and timing.
#[derive(Debug)]
pub struct Event {
    raw: hipEvent_t,
    device: Device,
}

// SAFETY: as for `Stream`; `Event` is `Send` but not `Sync` until HIP's
// rules for concurrent use are verified.
unsafe impl Send for Event {}

impl Event {
    /// Create an event on `device`.
    pub fn new(device: Device) -> Result<Self> {
        device.set_current()?;
        let api = hip()?;
        let f = api.hipEventCreate()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer for the call.
        check(api, "hipEventCreate", unsafe { f(&mut raw) })?;
        Ok(Self { raw, device })
    }

    /// Record the event after the work queued so far on `stream`.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipEventRecord()?;
        // SAFETY: both handles are live.
        check(api, "hipEventRecord", unsafe { f(self.raw, stream.raw) })
    }

    /// Wait until the event has happened.
    pub fn synchronize(&self) -> Result<()> {
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipEventSynchronize()?;
        // SAFETY: `self.raw` is a live event.
        check(api, "hipEventSynchronize", unsafe { f(self.raw) })
    }

    /// Whether the event has happened, without waiting.
    pub fn is_done(&self) -> Result<bool> {
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipEventQuery()?;
        // SAFETY: `self.raw` is a live event.
        check_ready(api, "hipEventQuery", unsafe { f(self.raw) })
    }

    /// Milliseconds from `start` to `self`; both must have been recorded and completed.
    pub fn elapsed_since(&self, start: &Event) -> Result<f32> {
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipEventElapsedTime()?;
        let mut ms = 0.0f32;
        // SAFETY: `ms` is a valid out-pointer and both events are live.
        check(api, "hipEventElapsedTime", unsafe {
            f(&mut ms, start.raw, self.raw)
        })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // A failure to destroy cannot be reported from `drop`.
        if self.device.set_current().is_err() {
            return;
        }
        if let Ok(f) = hip().and_then(|api| Ok(api.hipEventDestroy()?)) {
            // SAFETY: `self.raw` is a live event, destroyed exactly once here.
            let _ = unsafe { f(self.raw) };
        }
    }
}
