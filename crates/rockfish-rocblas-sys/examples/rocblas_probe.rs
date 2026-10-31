//! Loads rocBLAS, reports unresolved bound symbols and the library's own version string.
//! Pure query: creates no handle and touches no GPU.
fn main() {
    let api = match rockfish_rocblas_sys::RocblasApi::load() {
        Ok(api) => api,
        Err(e) => return println!("{e}"),
    };
    println!(
        "loaded from {:?}; unresolved: {:?}",
        api.library().loaded_from(),
        api.unresolved()
    );
    let size = api.rocblas_get_version_string_size().unwrap();
    let get = api.rocblas_get_version_string().unwrap();
    // SAFETY: `size` is the buffer length the library asked for.
    let (status, bytes) = unsafe {
        let mut n = 0usize;
        let size_status = size(&mut n);
        assert_eq!(size_status, 0);
        let mut buf = vec![0 as ::std::os::raw::c_char; n.max(1)];
        let status = get(buf.as_mut_ptr(), buf.len());
        (
            status,
            buf.iter()
                .take_while(|c| **c != 0)
                .map(|c| *c as u8)
                .collect::<Vec<u8>>(),
        )
    };
    println!(
        "status {status}, version string {:?}",
        String::from_utf8_lossy(&bytes)
    );
}
