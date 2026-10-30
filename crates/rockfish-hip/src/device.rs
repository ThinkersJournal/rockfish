//! Devices and the runtime version.

use rockfish_hip_sys::HipApi;

use crate::error::{Error, Result, check};
use crate::hip;

/// Free and total memory of a device, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Bytes currently free.
    pub free: usize,
    /// Bytes the device has.
    pub total: usize,
}

/// A HIP device, identified by its ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {
    ordinal: i32,
}

/// The HIP runtime version as `hipRuntimeGetVersion` reports it
/// (60450101 on the measured laptop).
pub fn runtime_version() -> Result<i32> {
    runtime_version_with(hip()?)
}

/// How many devices the runtime sees.
pub fn device_count() -> Result<u32> {
    device_count_with(hip()?)
}

fn runtime_version_with(api: &HipApi) -> Result<i32> {
    let f = api.hipRuntimeGetVersion()?;
    let mut v = 0;
    // SAFETY: `v` is a valid out-pointer for the call.
    check(api, "hipRuntimeGetVersion", unsafe { f(&mut v) })?;
    Ok(v)
}

fn device_count_with(api: &HipApi) -> Result<u32> {
    let f = api.hipGetDeviceCount()?;
    let mut n = 0;
    // SAFETY: `n` is a valid out-pointer for the call.
    check(api, "hipGetDeviceCount", unsafe { f(&mut n) })?;
    u32::try_from(n).map_err(|_| Error::Misuse("hipGetDeviceCount returned a negative count"))
}

impl Device {
    /// The device with this ordinal; an error if the runtime has no such device.
    pub fn get(ordinal: u32) -> Result<Self> {
        Self::get_with(hip()?, ordinal)
    }

    fn get_with(api: &HipApi, ordinal: u32) -> Result<Self> {
        if ordinal >= device_count_with(api)? {
            return Err(Error::Misuse("device ordinal out of range"));
        }
        let ordinal =
            i32::try_from(ordinal).map_err(|_| Error::Misuse("device ordinal too large"))?;
        Ok(Self { ordinal })
    }

    /// The ordinal.
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Make this device current on the calling thread (HIP keeps the
    /// current device per thread).
    pub fn set_current(&self) -> Result<()> {
        let api = hip()?;
        let f = api.hipSetDevice()?;
        // SAFETY: plain integer argument.
        check(api, "hipSetDevice", unsafe { f(self.ordinal as _) })
    }

    /// The device name, such as `AMD Radeon(TM) Graphics`.
    pub fn name(&self) -> Result<String> {
        let api = hip()?;
        let f = api.hipDeviceGetName()?;
        let mut buf = [0u8; 256];
        // SAFETY: `buf` is writable for the stated 256 bytes.
        check(api, "hipDeviceGetName", unsafe {
            f(buf.as_mut_ptr().cast(), buf.len() as _, self.ordinal as _)
        })?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    }

    /// Free and total memory. Makes this device current first.
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        self.set_current()?;
        let api = hip()?;
        let f = api.hipMemGetInfo()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both are valid out-pointers for the call.
        check(api, "hipMemGetInfo", unsafe { f(&mut free, &mut total) })?;
        Ok(MemoryInfo { free, total })
    }
}

#[cfg(all(test, any(windows, target_os = "linux")))]
mod tests {
    use super::*;
    use crate::test_support::not_hip;

    #[test]
    fn a_library_without_hip_reports_a_load_error_for_each_call() {
        let api = not_hip();
        assert!(matches!(device_count_with(&api), Err(Error::Load(_))));
        assert!(matches!(runtime_version_with(&api), Err(Error::Load(_))));
        assert!(matches!(Device::get_with(&api, 0), Err(Error::Load(_))));
    }

    // GPU tests: UNVERIFIED in CI (no GPU on the GitHub runners). They run
    // where HIP is installed and say so when it is not, never `#[ignore]`.
    #[test]
    fn device_queries_against_the_installed_runtime() {
        let Ok(api) = hip() else {
            eprintln!("no HIP runtime on this machine: GPU assertions not exercised");
            return;
        };
        let n = device_count_with(api).expect("count");
        assert!(runtime_version_with(api).expect("version") > 0);
        // Out of range is refused by the safe layer, before HIP is asked.
        assert!(matches!(Device::get_with(api, n), Err(Error::Misuse(_))));
        if n > 0 {
            let d = Device::get(0).expect("device 0");
            assert!(!d.name().expect("name").is_empty());
            let m = d.memory_info().expect("memory");
            assert!(m.total > 0 && m.free <= m.total);
        }
    }
}
