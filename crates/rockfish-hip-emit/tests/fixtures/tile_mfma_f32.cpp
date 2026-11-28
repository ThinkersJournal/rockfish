#include <hip/hip_runtime.h>

typedef float rf_v4f32 __attribute__((ext_vector_type(4)));

extern "C" __global__ void rockfish_hip_k_tile_mfma_f32(
    const float* __restrict__ p0,
    long long p0_s0,
    const float* __restrict__ p1,
    long long p1_s0,
    float* __restrict__ p2,
    long long p2_s0
) {
    const int rf_lane = (int)__lane_id();
    float t0 = {};
    float t1 = {};
    rf_v4f32 t2 = {};
    {
        const long long rf_epr = (long long)(((int)4LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        t0 = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + (rf_lane / 16)) * p0_s0];
    }
    {
        const long long rf_epr = (long long)(((int)16LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        t1 = p1[((long long)(((int)0LL)) + (rf_o0 + (rf_lane / 16)) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
    }
    {
        const float rf_v = 0.0f;
        t2[0] = rf_v;
        t2[1] = rf_v;
        t2[2] = rf_v;
        t2[3] = rf_v;
    }
    t2 = __builtin_amdgcn_mfma_f32_16x16x4f32(t0, t1, t2, 0, 0, 0);
    {
        const long long rf_epr = (long long)(((int)16LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        p2[((long long)(((int)0LL)) + (rf_o0 + (4 * (rf_lane / 16) + 0)) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[0];
        p2[((long long)(((int)0LL)) + (rf_o0 + (4 * (rf_lane / 16) + 1)) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[1];
        p2[((long long)(((int)0LL)) + (rf_o0 + (4 * (rf_lane / 16) + 2)) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[2];
        p2[((long long)(((int)0LL)) + (rf_o0 + (4 * (rf_lane / 16) + 3)) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[3];
    }
}
