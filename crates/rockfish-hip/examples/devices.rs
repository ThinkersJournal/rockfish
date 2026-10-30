//! Lists the HIP devices on this machine, or says why it cannot.

use rockfish_hip::{Device, DeviceBuffer, Error, Event, Stream, device_count, runtime_version};

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
        round_trip(device)?;
    }
    Ok(())
}

/// Sends 1 MiB to the device and back, checking it, and times an empty stream.
fn round_trip(device: Device) -> rockfish_hip::Result<()> {
    let sent: Vec<u8> = (0..1 << 20).map(|i| (i % 251) as u8).collect();
    let buffer = DeviceBuffer::from_slice(device, &sent)?;
    assert_eq!(
        buffer.to_vec()?,
        sent,
        "round trip returned different bytes"
    );
    let stream = Stream::new(device)?;
    let (start, end) = (Event::new(device)?, Event::new(device)?);
    start.record(&stream)?;
    end.record(&stream)?;
    end.synchronize()?;
    println!(
        "  round trip of {} bytes ok; empty stream took {} ms",
        buffer.len(),
        end.elapsed_since(&start)?
    );
    Ok(())
}
