// SPDX-License-Identifier: MIT OR Apache-2.0
// Written by Rockfish, not copied from AMD. A HOST emulation of a HIP thread block, for the
// differential in tests/differential.rs: one real host thread per HIP thread, the grid variables as
// (thread-local) variables, __syncthreads as a block barrier, and the subgroup operations
// (__shfl, __shfl_xor, __all, __any, __ballot) as exchanges through a slot array with a barrier per
// subgroup of SHIM_WAVE lanes. The subgroup semantics here are OUR MODEL of the hardware, written
// from AMD's HIP docs (rocm.docs.amd.com, "Warp shuffle functions", "Warp vote and ballot
// functions"): a source lane is taken modulo `width`; an XOR source past the width returns the
// calling lane's own value; __ballot returns 64 bits. No device has confirmed any of it: UNMEASURED.
#pragma once
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#ifndef SHIM_WAVE
#define SHIM_WAVE 32
#endif
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
struct hip_dim3_shim { unsigned x, y, z; };
inline thread_local hip_dim3_shim threadIdx{0, 0, 0};
inline thread_local hip_dim3_shim blockIdx{0, 0, 0};
inline hip_dim3_shim blockDim{1, 1, 1};
inline hip_dim3_shim gridDim{1, 1, 1};

struct shim_barrier {
  std::mutex m;
  std::condition_variable cv;
  unsigned count = 0, gen = 0, n = 1;
  void reset(unsigned n_) { n = n_; count = 0; }
  void wait() {
    std::unique_lock<std::mutex> l(m);
    unsigned g = gen;
    if (++count == n) { count = 0; ++gen; cv.notify_all(); }
    else cv.wait(l, [&] { return g != gen; });
  }
};
inline shim_barrier g_block;
inline shim_barrier g_sub[64];
inline unsigned long long g_slot[2048];

inline unsigned shim_tid() { return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; }
inline void __syncthreads() { g_block.wait(); }

template <class T> T __shfl(T v, int srcLane, int width = SHIM_WAVE) {
  unsigned t = shim_tid(), sub = t / SHIM_WAVE, lane = t % SHIM_WAVE;
  unsigned long long b = 0; std::memcpy(&b, &v, sizeof(T));
  g_slot[t] = b; g_sub[sub].wait();
  int s = ((srcLane % width) + width) % width;
  unsigned long long r = g_slot[sub * SHIM_WAVE + (lane / width) * width + s];
  g_sub[sub].wait();
  T out; std::memcpy(&out, &r, sizeof(T)); return out;
}
template <class T> T __shfl_xor(T v, int laneMask, int width = SHIM_WAVE) {
  unsigned t = shim_tid(), sub = t / SHIM_WAVE, lane = t % SHIM_WAVE;
  unsigned long long b = 0; std::memcpy(&b, &v, sizeof(T));
  g_slot[t] = b; g_sub[sub].wait();
  int inw = (int)(lane % width), s = inw ^ laneMask;
  if (s < 0 || s >= width) s = inw;  // past the width: the caller's own value
  unsigned long long r = g_slot[sub * SHIM_WAVE + (lane / width) * width + s];
  g_sub[sub].wait();
  T out; std::memcpy(&out, &r, sizeof(T)); return out;
}
inline unsigned long long shim_vote_bits(int predicate) {
  unsigned t = shim_tid(), sub = t / SHIM_WAVE;
  g_slot[t] = predicate != 0 ? 1ull : 0ull; g_sub[sub].wait();
  unsigned long long m = 0;
  for (unsigned i = 0; i < SHIM_WAVE; ++i) m |= (g_slot[sub * SHIM_WAVE + i] & 1ull) << i;
  g_sub[sub].wait();
  return m;
}
inline int __all(int predicate) { unsigned long long m = shim_vote_bits(predicate); return m == (SHIM_WAVE == 64 ? ~0ull : ((1ull << SHIM_WAVE) - 1)) ? 1 : 0; }
inline int __any(int predicate) { return shim_vote_bits(predicate) != 0 ? 1 : 0; }
inline unsigned long long __ballot(int predicate) { return shim_vote_bits(predicate); }

// Run `kernel` once per thread of every block, blocks one after another.
template <class F> void shim_launch(unsigned bx, unsigned by, unsigned bz, unsigned gx, unsigned gy, unsigned gz, F kernel) {
  blockDim = {bx, by, bz}; gridDim = {gx, gy, gz};
  unsigned n = bx * by * bz;
  for (unsigned z = 0; z < gz; ++z) for (unsigned y = 0; y < gy; ++y) for (unsigned x = 0; x < gx; ++x) {
    g_block.reset(n);
    for (unsigned s = 0; s < n / SHIM_WAVE + 1 && s < 64; ++s) g_sub[s].reset(SHIM_WAVE);
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < n; ++t) ts.emplace_back([=] {
      threadIdx = {t % bx, (t / bx) % by, t / (bx * by)};
      blockIdx = {x, y, z};
      kernel();
    });
    for (auto& th : ts) th.join();
  }
}
