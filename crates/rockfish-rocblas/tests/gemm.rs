//! Handle and GEMM against a real rocBLAS and device when both exist.
//!
//! With no rocBLAS (the usual state of the Windows laptop: Ollama's bundle is
//! not on `PATH`) each test prints that it skipped and passes; with one, any
//! failure is a failure. A pass here with a "skipped" line proves nothing about
//! GEMM. Real GEMM correctness is a rig row.

use rockfish_hip::{Device, DeviceBuffer, Stream};
use rockfish_rocblas::{Error, Handle, Op};

/// `None` (and a printed reason) when rocBLAS or HIP or a device is absent.
fn device() -> Option<Device> {
    match Device::get(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipped: no HIP device ({e})");
            None
        }
    }
}

#[test]
fn handle_creation_is_a_load_error_without_rocblas() {
    let Some(device) = device() else { return };
    match Handle::new(device) {
        Err(Error::Load(e)) => eprintln!("skipped: no rocBLAS ({e})"),
        Ok(h) => {
            assert_eq!(h.device(), device);
            h.synchronize()
                .expect("an idle default stream synchronizes");
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[test]
fn gemm_arguments_are_refused_before_rocblas() {
    let Some(device) = device() else { return };
    let Ok(handle) = Handle::new(device) else {
        eprintln!("skipped: no rocBLAS handle");
        return;
    };
    let a = DeviceBuffer::<f32>::alloc(device, 4).unwrap();
    let b = DeviceBuffer::<f32>::alloc(device, 4).unwrap();
    let mut c = DeviceBuffer::<f32>::alloc(device, 3).unwrap();
    // C needs 4 elements for a 2x2 result.
    let r = handle.gemm(
        Op::None,
        Op::None,
        2,
        2,
        2,
        1.0,
        &a,
        2,
        &b,
        2,
        0.0,
        &mut c,
        2,
    );
    assert!(matches!(r, Err(Error::Misuse(_))), "{r:?}");
}

#[test]
fn f32_gemm_matches_a_host_product() {
    let Some(device) = device() else { return };
    let stream = match Stream::new(device) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skipped: no stream ({e})");
            return;
        }
    };
    let handle = match Handle::on_stream(&stream) {
        Ok(h) => h,
        Err(Error::Load(e)) => {
            eprintln!("skipped: no rocBLAS ({e})");
            return;
        }
        Err(e) => panic!("unexpected error: {e}"),
    };
    // Column-major: A is 2x3, B is 3x2, C = 2*A*B + 1*C0.
    let a_host = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // [[1,2,3],[4,5,6]]
    let b_host = [7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0]; // [[7,8],[9,10],[11,12]]
    let c0 = [1.0f32, 1.0, 1.0, 1.0];
    let a = DeviceBuffer::from_slice(device, &a_host).unwrap();
    let b = DeviceBuffer::from_slice(device, &b_host).unwrap();
    let mut c = DeviceBuffer::from_slice(device, &c0).unwrap();
    handle
        .gemm(
            Op::None,
            Op::None,
            2,
            2,
            3,
            2.0,
            &a,
            2,
            &b,
            3,
            1.0,
            &mut c,
            2,
        )
        .expect("gemm");
    handle.synchronize().expect("synchronize");
    // A*B = [[58,64],[139,154]]; column-major [58,139,64,154].
    let expect = [117.0f32, 279.0, 129.0, 309.0];
    assert_eq!(c.to_vec().unwrap(), expect);
}
