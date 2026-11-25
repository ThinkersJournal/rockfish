// SPDX-License-Identifier: MIT OR Apache-2.0
// Written by Rockfish, not copied from AMD. A host stand-in for <hip/hip_runtime.h> that gives a C++
// compiler the names the emitted kernels use, so `tests/syntax.rs` can run `-fsyntax-only` over
// them. It checks syntax and typing of the emitted text only: no HIP semantics, no device code.
#pragma once
#include <cmath>
#include <cstdint>
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
struct hip_dim3_shim { unsigned x, y, z; };
static hip_dim3_shim blockIdx, blockDim, threadIdx, gridDim;
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct double2 { double x, y; };
struct double4 { double x, y, z, w; };
struct ushort2 { unsigned short x, y; };
struct ushort4 { unsigned short x, y, z, w; };
