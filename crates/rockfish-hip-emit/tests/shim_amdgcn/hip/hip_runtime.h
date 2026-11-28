// SPDX-License-Identifier: MIT OR Apache-2.0
// Written by Rockfish, not copied from AMD. A stand-in for <hip/hip_runtime.h> for `tests/tile_syntax.rs`, which compiles
// emitted tile kernels with clang's AMDGPU target (`-x c++ -S`, no HIP runtime, no device). It gives the names the emitted
// text uses: `__global__` as an AMDGPU kernel and `__lane_id()` as the mbcnt pair HIP's own `__lane_id` is built from.
// Syntax, typing and instruction selection only: no HIP semantics.
#pragma once
#define __global__ __attribute__((amdgpu_kernel))
#define __device__
#define __host__
#define __shared__
static inline __attribute__((always_inline)) unsigned int __lane_id() {
    return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}
