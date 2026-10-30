//! Lists the HIP devices on this machine, or says why it cannot.

use rockfish_hip::{Device, Error, device_count, runtime_version};

fn main() {
    match run() {
        Ok(()) => {}
        Err(Error::Load(e)) => println!("no usable HIP runtime: {e}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

fn run() -> rockfish_hip::Result<()> {
    println!("HIP runtime version {}", runtime_version()?);
    for ordinal in 0..device_count()? {
        let device = Device::get(ordinal)?;
        let memory = device.memory_info()?;
        println!(
            "device {}: {} ({} of {} bytes free)",
            device.ordinal(),
            device.name()?,
            memory.free,
            memory.total
        );
    }
    Ok(())
}
