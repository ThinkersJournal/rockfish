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

// Subgroup shuffles and votes: signatures as AMD's HIP docs state them (rocm.docs.amd.com, HIP
// reference "Warp shuffle functions", "Warp vote and ballot functions"), identity bodies: typing
// only, no semantics. __ballot returns 64 bits on both wave sizes, as documented.
template <class T> T __shfl(T var, int srcLane, int width = 64) { (void)srcLane; (void)width; return var; }
template <class T> T __shfl_xor(T var, int laneMask, int width = 64) { (void)laneMask; (void)width; return var; }
inline int __all(int predicate) { return predicate; }
inline int __any(int predicate) { return predicate; }
inline unsigned long long __ballot(int predicate) { return (unsigned long long)predicate; }
