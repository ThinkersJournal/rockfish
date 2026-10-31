//! Code objects, kernels and launches.

use std::ffi::{CStr, c_void};
use std::marker::PhantomData;

use rockfish_hip_sys::{hipFunction_t, hipModule_t};

use crate::buffer::DeviceBuffer;
use crate::device::Device;
use crate::error::{Error, Result, check};
use crate::hip;
use crate::repr::DeviceRepr;
use crate::stream::Stream;

/// A loaded device code object (an ELF image built for this GPU's architecture).
#[derive(Debug)]
pub struct Module {
    raw: hipModule_t,
    device: Device,
}

// SAFETY: the handle may be used from any thread; each call makes the
// module's device current first. Concurrent use is unverified, so `Module`
// is `Send` but not `Sync`.
unsafe impl Send for Module {}

impl Module {
    /// Load a code object already in memory onto `device`.
    ///
    /// `image` must be a code object for the device's architecture, such as
    /// the output of `hipcc --genco`. The runtime is handed a pointer only,
    /// so a corrupt image is the runtime's to reject; it is reported as an
    /// [`Error::Api`].
    pub fn from_code_object(device: Device, image: &[u8]) -> Result<Self> {
        if image.is_empty() {
            return Err(Error::Misuse("empty code object"));
        }
        device.set_current()?;
        let api = hip()?;
        let f = api.hipModuleLoadData()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer and `image` is readable.
        check(api, "hipModuleLoadData", unsafe {
            f(&mut raw, image.as_ptr().cast())
        })?;
        Ok(Self { raw, device })
    }

    /// The device this module is loaded on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Look up a kernel by its (mangled) symbol name.
    pub fn function(&self, name: &CStr) -> Result<Function<'_>> {
        self.device.set_current()?;
        let api = hip()?;
        let f = api.hipModuleGetFunction()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer, `self.raw` a live module and
        // `name` NUL-terminated.
        check(api, "hipModuleGetFunction", unsafe {
            f(&mut raw, self.raw, name.as_ptr())
        })?;
        Ok(Function { raw, module: self })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // A failure to unload cannot be reported from `drop`.
        if self.device.set_current().is_err() {
            return;
        }
        if let Ok(f) = hip().and_then(|api| Ok(api.hipModuleUnload()?)) {
            // SAFETY: `self.raw` is a live module, unloaded exactly once here.
            let _ = unsafe { f(self.raw) };
        }
    }
}

/// A kernel inside a [`Module`]; borrows the module so it cannot outlive it.
#[derive(Debug, Clone, Copy)]
pub struct Function<'m> {
    raw: hipFunction_t,
    module: &'m Module,
}

/// Grid and block shape of a launch, in blocks and threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Blocks per grid in x, y, z.
    pub grid: (u32, u32, u32),
    /// Threads per block in x, y, z.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory per block, in bytes.
    pub shared_bytes: u32,
}

impl LaunchConfig {
    /// A one-dimensional launch of `blocks` blocks of `threads` threads.
    pub fn linear(blocks: u32, threads: u32) -> Self {
        Self {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_bytes: 0,
        }
    }

    fn validate(&self) -> Result<()> {
        let dims = [
            self.grid.0,
            self.grid.1,
            self.grid.2,
            self.block.0,
            self.block.1,
            self.block.2,
        ];
        if dims.contains(&0) {
            return Err(Error::Misuse("launch dimensions must be non-zero"));
        }
        Ok(())
    }
}

/// Kernel arguments, in the kernel's parameter order.
///
/// Each argument is borrowed until the launch call returns, so the addresses
/// HIP reads stay valid. A buffer is passed as its device pointer.
#[derive(Debug, Default)]
pub struct KernelArgs<'a> {
    slots: Vec<*mut c_void>,
    _borrow: PhantomData<&'a ()>,
}

impl<'a> KernelArgs<'a> {
    /// No arguments yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a plain value, passed by value to the kernel.
    pub fn value<T: DeviceRepr>(mut self, value: &'a T) -> Self {
        self.slots.push(std::ptr::from_ref(value).cast_mut().cast());
        self
    }

    /// Append a device buffer, passed as a pointer to its first element.
    pub fn buffer<T: DeviceRepr>(mut self, buffer: &'a DeviceBuffer<T>) -> Self {
        self.slots.push(buffer.ptr_slot());
        self
    }

    /// Number of arguments so far.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether there are no arguments.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl Function<'_> {
    /// Queue the kernel on `stream` and return without waiting.
    ///
    /// # Safety
    ///
    /// HIP cannot check arguments against the kernel's signature. `args` must
    /// match the kernel's parameters in number, order, size and meaning, and
    /// every buffer must be large enough for what the kernel reads and writes.
    /// The buffers and `stream` must stay alive until the work has finished.
    pub unsafe fn launch(
        &self,
        config: LaunchConfig,
        stream: &Stream,
        args: &KernelArgs<'_>,
    ) -> Result<()> {
        config.validate()?;
        if stream.device() != self.module.device {
            return Err(Error::Misuse("stream and module are on different devices"));
        }
        self.module.device.set_current()?;
        let api = hip()?;
        let f = api.hipModuleLaunchKernel()?;
        // SAFETY: the caller upholds the contract above; `args.slots` is an
        // array of pointers to live values, which HIP only reads during the call.
        check(api, "hipModuleLaunchKernel", unsafe {
            f(
                self.raw,
                config.grid.0,
                config.grid.1,
                config.grid.2,
                config.block.0,
                config.block.1,
                config.block.2,
                config.shared_bytes,
                stream.as_raw(),
                args.slots.as_ptr().cast_mut(),
                std::ptr::null_mut(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_dimensions_are_refused() {
        assert!(LaunchConfig::linear(1, 64).validate().is_ok());
        assert!(matches!(
            LaunchConfig::linear(0, 64).validate(),
            Err(Error::Misuse(_))
        ));
        let mut c = LaunchConfig::linear(1, 64);
        c.block.2 = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn args_collect_in_order_and_point_at_the_values() {
        let (a, b) = (7u32, 9u64);
        let args = KernelArgs::new().value(&a).value(&b);
        assert_eq!(args.len(), 2);
        assert_eq!(args.slots[0], std::ptr::from_ref(&a).cast_mut().cast());
        assert_eq!(args.slots[1], std::ptr::from_ref(&b).cast_mut().cast());
        assert!(KernelArgs::new().is_empty());
    }

    #[test]
    fn empty_code_object_is_refused_before_hip_is_asked() {
        // Needs a Device value, which needs no HIP call to construct only on
        // machines with a device; the empty check precedes every HIP call.
        let Some(device) = device() else { return };
        assert!(matches!(
            Module::from_code_object(device, &[]),
            Err(Error::Misuse(_))
        ));
    }

    // GPU tests: UNVERIFIED in CI (no GPU on the GitHub runners). Never
    // `#[ignore]`: without HIP they say so and stop.
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
    fn garbage_code_object_is_an_api_error_with_a_name() {
        let Some(device) = device() else { return };
        match Module::from_code_object(device, &[0xAB; 256]) {
            Err(Error::Api { call, name, .. }) => {
                assert_eq!(call, "hipModuleLoadData");
                eprintln!("garbage image reported as {name}");
                assert!(name.starts_with("hipError"), "name was {name}");
            }
            other => panic!("expected an Api error, got {:?}", other.map(|_| ())),
        }
    }

    /// Launch a real kernel. Needs a code object built for this GPU, which
    /// this crate cannot compile; point `ROCKFISH_TEST_CODE_OBJECT` at one
    /// containing `rockfish_add_one(unsigned* data, unsigned n)` (adds 1 to
    /// each of the first `n` elements). Without it the test says so and
    /// stops: the launch path is UNVERIFIED until one runs.
    #[test]
    fn add_one_kernel_from_an_external_code_object() {
        let Some(device) = device() else { return };
        let Ok(path) = std::env::var("ROCKFISH_TEST_CODE_OBJECT") else {
            eprintln!("ROCKFISH_TEST_CODE_OBJECT unset: kernel launch not exercised");
            return;
        };
        let image = std::fs::read(&path).expect("read code object");
        let module = Module::from_code_object(device, &image).expect("load");
        let kernel = module.function(c"rockfish_add_one").expect("symbol");
        let n = 1000u32;
        let host: Vec<u32> = (0..n).map(|i| i * 3).collect();
        let buffer = DeviceBuffer::from_slice(device, &host).expect("upload");
        let stream = Stream::new(device).expect("stream");
        let args = KernelArgs::new().buffer(&buffer).value(&n);
        // SAFETY: matches `rockfish_add_one(unsigned*, unsigned)`; buffer and
        // stream outlive the synchronize below.
        unsafe { kernel.launch(LaunchConfig::linear(n.div_ceil(64), 64), &stream, &args) }
            .expect("launch");
        stream.synchronize().expect("sync");
        let expected: Vec<u32> = host.iter().map(|v| v + 1).collect();
        assert_eq!(buffer.to_vec().expect("download"), expected);
    }
}
