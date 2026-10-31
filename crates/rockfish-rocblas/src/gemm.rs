//! Typed GEMM over `rocblas_gemm_ex` and `rocblas_gemm_strided_batched_ex`.
//!
//! Matrices are column-major, as in BLAS: element `(i, j)` of a matrix with
//! leading dimension `ld` is at `i + j * ld`. `C = alpha * op(A) * op(B) + beta * C`
//! with `op(A)` m by k, `op(B)` k by n and `C` m by n. Every element type goes
//! through the `_ex` entry points (one code path); `f16` and `bf16` accumulate
//! in `f32`, and `alpha`/`beta` are `f32` for them.

use std::ffi::c_void;

use rockfish_hip::{DeviceBuffer, DeviceRepr};
use rockfish_rocblas_sys::{
    rocblas_datatype__rocblas_datatype_f32_r, rocblas_datatype__rocblas_datatype_f64_r,
    rocblas_gemm_algo__rocblas_gemm_algo_standard,
    rocblas_operation__rocblas_operation_conjugate_transpose,
    rocblas_operation__rocblas_operation_none, rocblas_operation__rocblas_operation_transpose,
};

use crate::error::{Error, Result, check};
use crate::handle::Handle;
use crate::rocblas;

/// Whether a matrix is used as stored, transposed, or conjugate-transposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Use the matrix as stored.
    None,
    /// Use its transpose.
    Transpose,
    /// Use its conjugate transpose (the same as `Transpose` for real types).
    ConjugateTranspose,
}

impl Op {
    fn raw(self) -> i32 {
        match self {
            Op::None => rocblas_operation__rocblas_operation_none,
            Op::Transpose => rocblas_operation__rocblas_operation_transpose,
            Op::ConjugateTranspose => rocblas_operation__rocblas_operation_conjugate_transpose,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An element type GEMM is wired for: `f32`, `f64`, and with feature `half`
/// `half::f16` and `half::bf16`. Sealed.
pub trait GemmElement: DeviceRepr + sealed::Sealed {
    /// The type of `alpha` and `beta`: `f32` for `f32`, `f16` and `bf16`, else `f64`.
    type Scalar: Copy;
    #[doc(hidden)]
    const DATATYPE: i32;
    #[doc(hidden)]
    const COMPUTE: i32;
}

macro_rules! element {
    ($t:ty, $scalar:ty, $data:expr, $compute:expr) => {
        impl sealed::Sealed for $t {}
        impl GemmElement for $t {
            type Scalar = $scalar;
            const DATATYPE: i32 = $data;
            const COMPUTE: i32 = $compute;
        }
    };
}

element!(
    f32,
    f32,
    rocblas_datatype__rocblas_datatype_f32_r,
    rocblas_datatype__rocblas_datatype_f32_r
);
element!(
    f64,
    f64,
    rocblas_datatype__rocblas_datatype_f64_r,
    rocblas_datatype__rocblas_datatype_f64_r
);
#[cfg(feature = "half")]
element!(
    half::f16,
    f32,
    rockfish_rocblas_sys::rocblas_datatype__rocblas_datatype_f16_r,
    rocblas_datatype__rocblas_datatype_f32_r
);
#[cfg(feature = "half")]
element!(
    half::bf16,
    f32,
    rockfish_rocblas_sys::rocblas_datatype__rocblas_datatype_bf16_r,
    rocblas_datatype__rocblas_datatype_f32_r
);

/// Shape arguments converted to what rocBLAS takes, with the buffer lengths
/// they require (in elements).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub a_len: usize,
    pub b_len: usize,
    pub c_len: usize,
}

fn dim(v: usize, what: &'static str) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::Misuse(what))
}

/// Elements a `rows` by `cols` column-major matrix with leading dimension
/// `ld` spans in memory.
fn span(rows: usize, cols: usize, ld: usize, what: &'static str) -> Result<usize> {
    if ld < rows.max(1) {
        return Err(Error::Misuse(what));
    }
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    (cols - 1)
        .checked_mul(ld)
        .and_then(|v| v.checked_add(rows))
        .ok_or(Error::Misuse("matrix size overflows"))
}

/// Check the shape arguments of one GEMM and work out the buffer lengths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan(
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) -> Result<Plan> {
    let (a_rows, a_cols) = if op_a == Op::None { (m, k) } else { (k, m) };
    let (b_rows, b_cols) = if op_b == Op::None { (k, n) } else { (n, k) };
    Ok(Plan {
        m: dim(m, "m does not fit in 32 bits")?,
        n: dim(n, "n does not fit in 32 bits")?,
        k: dim(k, "k does not fit in 32 bits")?,
        lda: dim(lda, "lda does not fit in 32 bits")?,
        ldb: dim(ldb, "ldb does not fit in 32 bits")?,
        ldc: dim(ldc, "ldc does not fit in 32 bits")?,
        a_len: span(
            a_rows,
            a_cols,
            lda,
            "lda is smaller than the rows of A as stored",
        )?,
        b_len: span(
            b_rows,
            b_cols,
            ldb,
            "ldb is smaller than the rows of B as stored",
        )?,
        c_len: span(m, n, ldc, "ldc is smaller than m")?,
    })
}

/// Elements `batch` matrices spanning `single` each, `stride` apart, need.
fn batched_len(single: usize, stride: usize, batch: usize) -> Result<usize> {
    if batch == 0 || single == 0 {
        return Ok(0);
    }
    (batch - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(single))
        .ok_or(Error::Misuse("batch size overflows"))
}

fn long(v: usize, what: &'static str) -> Result<i64> {
    i64::try_from(v).map_err(|_| Error::Misuse(what))
}

fn need<T: DeviceRepr>(buf: &DeviceBuffer<T>, len: usize, what: &'static str) -> Result<()> {
    if buf.len() < len {
        Err(Error::Misuse(what))
    } else {
        Ok(())
    }
}

impl Handle<'_> {
    fn same_device<T: DeviceRepr>(&self, buf: &DeviceBuffer<T>) -> Result<()> {
        if buf.device() == self.device {
            Ok(())
        } else {
            Err(Error::Misuse(
                "buffer is on a different device than the handle",
            ))
        }
    }

    /// `C = alpha * op(A) * op(B) + beta * C`, queued on the handle's stream.
    ///
    /// `op(A)` is `m` by `k`, `op(B)` is `k` by `n`, `C` is `m` by `n`, all
    /// column-major with leading dimensions `lda`, `ldb`, `ldc`. The buffers
    /// must be on the handle's device and long enough for their matrix.
    /// Returns before the work finishes; see [`Handle::synchronize`].
    ///
    /// UNVERIFIED on a GPU: no GEMM has run on any machine measured.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm<T: GemmElement>(
        &self,
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: T::Scalar,
        a: &DeviceBuffer<T>,
        lda: usize,
        b: &DeviceBuffer<T>,
        ldb: usize,
        beta: T::Scalar,
        c: &mut DeviceBuffer<T>,
        ldc: usize,
    ) -> Result<()> {
        let p = plan(op_a, op_b, m, n, k, lda, ldb, ldc)?;
        need(a, p.a_len, "buffer A is too short for m, k and lda")?;
        need(b, p.b_len, "buffer B is too short for k, n and ldb")?;
        need(c, p.c_len, "buffer C is too short for m, n and ldc")?;
        self.same_device(a)?;
        self.same_device(b)?;
        self.same_device(c)?;
        self.device.set_current()?;
        let api = rocblas()?;
        let f = api.rocblas_gemm_ex()?;
        let c_ptr = c.as_raw_ptr();
        // SAFETY: the handle is live; the buffers are live device allocations
        // on the handle's device, checked above to hold every element the call
        // touches; `alpha`/`beta` are host scalars (the handle is in host
        // pointer mode) of `T::Scalar`, the type `T::COMPUTE` names. C is both
        // input and output, so the same pointer is passed for `c` and `d`.
        let code = unsafe {
            f(
                self.raw,
                op_a.raw(),
                op_b.raw(),
                p.m,
                p.n,
                p.k,
                std::ptr::from_ref(&alpha).cast::<c_void>(),
                a.as_raw_ptr().cast_const(),
                T::DATATYPE,
                p.lda,
                b.as_raw_ptr().cast_const(),
                T::DATATYPE,
                p.ldb,
                std::ptr::from_ref(&beta).cast::<c_void>(),
                c_ptr.cast_const(),
                T::DATATYPE,
                p.ldc,
                c_ptr,
                T::DATATYPE,
                p.ldc,
                T::COMPUTE,
                rocblas_gemm_algo__rocblas_gemm_algo_standard,
                0,
                0,
            )
        };
        check(api, "rocblas_gemm_ex", code)
    }

    /// `batch` GEMMs in one call: matrix `i` of each operand starts
    /// `i * stride` elements after matrix 0. Strides are in elements and may
    /// be 0 for `A` and `B` (one matrix broadcast over the batch); the stride
    /// of `C` must be at least the span of one `C` matrix, so results do not
    /// overlap. Otherwise as [`Handle::gemm`].
    ///
    /// UNVERIFIED on a GPU: no GEMM has run on any machine measured.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_strided_batched<T: GemmElement>(
        &self,
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: T::Scalar,
        a: &DeviceBuffer<T>,
        lda: usize,
        stride_a: usize,
        b: &DeviceBuffer<T>,
        ldb: usize,
        stride_b: usize,
        beta: T::Scalar,
        c: &mut DeviceBuffer<T>,
        ldc: usize,
        stride_c: usize,
        batch: usize,
    ) -> Result<()> {
        let p = plan(op_a, op_b, m, n, k, lda, ldb, ldc)?;
        if batch > 1 && stride_c < p.c_len {
            return Err(Error::Misuse("stride of C is smaller than one C matrix"));
        }
        need(
            a,
            batched_len(p.a_len, stride_a, batch)?,
            "buffer A is too short for the batch",
        )?;
        need(
            b,
            batched_len(p.b_len, stride_b, batch)?,
            "buffer B is too short for the batch",
        )?;
        need(
            c,
            batched_len(p.c_len, stride_c, batch)?,
            "buffer C is too short for the batch",
        )?;
        self.same_device(a)?;
        self.same_device(b)?;
        self.same_device(c)?;
        let batch_count = dim(batch, "batch does not fit in 32 bits")?;
        let (sa, sb, sc) = (
            long(stride_a, "stride of A does not fit in 63 bits")?,
            long(stride_b, "stride of B does not fit in 63 bits")?,
            long(stride_c, "stride of C does not fit in 63 bits")?,
        );
        self.device.set_current()?;
        let api = rocblas()?;
        let f = api.rocblas_gemm_strided_batched_ex()?;
        let c_ptr = c.as_raw_ptr();
        // SAFETY: as for `gemm`; the batched lengths checked above cover every
        // matrix the call touches.
        let code = unsafe {
            f(
                self.raw,
                op_a.raw(),
                op_b.raw(),
                p.m,
                p.n,
                p.k,
                std::ptr::from_ref(&alpha).cast::<c_void>(),
                a.as_raw_ptr().cast_const(),
                T::DATATYPE,
                p.lda,
                sa,
                b.as_raw_ptr().cast_const(),
                T::DATATYPE,
                p.ldb,
                sb,
                std::ptr::from_ref(&beta).cast::<c_void>(),
                c_ptr.cast_const(),
                T::DATATYPE,
                p.ldc,
                sc,
                c_ptr,
                T::DATATYPE,
                p.ldc,
                sc,
                batch_count,
                T::COMPUTE,
                rocblas_gemm_algo__rocblas_gemm_algo_standard,
                0,
                0,
            )
        };
        check(api, "rocblas_gemm_strided_batched_ex", code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn misuse<T: std::fmt::Debug>(r: Result<T>) -> &'static str {
        match r {
            Err(Error::Misuse(m)) => m,
            other => panic!("expected Misuse, got {other:?}"),
        }
    }

    #[test]
    fn plan_sizes_a_plain_product() {
        // A is 3x4, B is 4x5, C is 3x5, tight leading dimensions.
        let p = plan(Op::None, Op::None, 3, 5, 4, 3, 4, 3).unwrap();
        assert_eq!((p.a_len, p.b_len, p.c_len), (12, 20, 15));
    }

    #[test]
    fn plan_swaps_the_stored_shape_under_transpose() {
        // op(A) = A^T is 3x4, so A is stored 4x3.
        let p = plan(Op::Transpose, Op::ConjugateTranspose, 3, 5, 4, 4, 5, 3).unwrap();
        assert_eq!((p.a_len, p.b_len), (12, 20));
        // The same lda is too small once A is stored 4 rows tall.
        assert!(plan(Op::Transpose, Op::None, 3, 5, 4, 3, 4, 3).is_err());
    }

    #[test]
    fn plan_counts_padding_between_columns_but_not_after_the_last() {
        // 2x3 matrix with ld 5: columns start at 0, 5, 10; last element at 11.
        assert_eq!(span(2, 3, 5, "x").unwrap(), 12);
    }

    #[test]
    fn plan_rejects_a_small_leading_dimension() {
        let m = misuse(plan(Op::None, Op::None, 3, 5, 4, 2, 4, 3));
        assert!(m.contains("lda"), "{m}");
        let m = misuse(plan(Op::None, Op::None, 3, 5, 4, 3, 3, 3));
        assert!(m.contains("ldb"), "{m}");
        let m = misuse(plan(Op::None, Op::None, 3, 5, 4, 3, 4, 2));
        assert!(m.contains("ldc"), "{m}");
        // Control: the same call with the leading dimensions fixed passes.
        assert!(plan(Op::None, Op::None, 3, 5, 4, 3, 4, 3).is_ok());
    }

    #[test]
    fn plan_requires_ld_of_at_least_one_even_when_empty() {
        assert!(plan(Op::None, Op::None, 0, 0, 0, 1, 1, 1).is_ok());
        assert!(plan(Op::None, Op::None, 0, 0, 0, 0, 1, 1).is_err());
        let p = plan(Op::None, Op::None, 0, 7, 3, 1, 3, 1).unwrap();
        assert_eq!((p.a_len, p.c_len), (0, 0));
    }

    #[test]
    fn plan_rejects_sizes_that_do_not_fit_rocblas() {
        let big = i32::MAX as usize + 1;
        assert!(plan(Op::None, Op::None, big, 1, 1, big, 1, big).is_err());
        assert!(plan(Op::None, Op::None, 1, 1, 1, big, 1, 1).is_err());
        assert!(plan(Op::None, Op::None, 1, 1, 1, i32::MAX as usize, 1, 1).is_ok());
    }

    #[test]
    fn plan_reports_overflow_instead_of_wrapping() {
        let r = span(2, usize::MAX, 2, "x");
        assert!(matches!(r, Err(Error::Misuse("matrix size overflows"))));
    }

    #[test]
    fn batched_len_counts_strides_between_matrices_only() {
        assert_eq!(batched_len(12, 12, 3).unwrap(), 36);
        assert_eq!(batched_len(12, 20, 3).unwrap(), 52);
        // Broadcast: stride 0 needs one matrix.
        assert_eq!(batched_len(12, 0, 8).unwrap(), 12);
        assert_eq!(batched_len(12, 12, 0).unwrap(), 0);
        assert!(batched_len(12, usize::MAX, 3).is_err());
    }

    #[test]
    fn element_tables_match_the_header_values() {
        assert_eq!(<f32 as GemmElement>::DATATYPE, 151);
        assert_eq!(<f64 as GemmElement>::DATATYPE, 152);
        assert_eq!(<f64 as GemmElement>::COMPUTE, 152);
        assert_eq!(Op::None.raw(), 111);
        assert_eq!(Op::Transpose.raw(), 112);
        assert_eq!(Op::ConjugateTranspose.raw(), 113);
    }

    #[cfg(feature = "half")]
    #[test]
    fn half_types_accumulate_in_f32() {
        assert_eq!(<half::f16 as GemmElement>::DATATYPE, 150);
        assert_eq!(<half::bf16 as GemmElement>::DATATYPE, 168);
        assert_eq!(<half::f16 as GemmElement>::COMPUTE, 151);
        assert_eq!(<half::bf16 as GemmElement>::COMPUTE, 151);
    }
}
