//! Prints what the HIP runtime on this machine reports. Run:
//! `cargo run -p rockfish-hip-sys --example probe`

use rockfish_hip_sys::HipApi;

fn main() {
    let api = match HipApi::load() {
        Ok(api) => api,
        Err(error) => {
            println!("no HIP runtime: {error}");
            return;
        }
    };
    println!("loaded from {:?}", api.library().loaded_from());

    let mut version = 0;
    let mut count = 0;
    // SAFETY: both pointers are valid out-parameters for the calls.
    unsafe {
        let status = api.hipRuntimeGetVersion().expect("hipRuntimeGetVersion")(&mut version);
        println!("hipRuntimeGetVersion status {status}, version {version}");
        let status = api.hipGetDeviceCount().expect("hipGetDeviceCount")(&mut count);
        println!("hipGetDeviceCount status {status}, count {count}");
        let mut name = [0 as std::os::raw::c_char; 256];
        for device in 0..count {
            let status = api.hipDeviceGetName().expect("hipDeviceGetName")(
                name.as_mut_ptr(),
                name.len() as i32,
                device,
            );
            let text = std::ffi::CStr::from_ptr(name.as_ptr()).to_string_lossy();
            println!("device {device}: status {status}, {text}");
        }
    }
}
