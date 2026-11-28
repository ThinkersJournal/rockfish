#include <hip/hip_runtime.h>

typedef _Float16 rf_v16f16 __attribute__((ext_vector_type(16)));
typedef float rf_v8f32 __attribute__((ext_vector_type(8)));

extern "C" __global__ void rockfish_hip_k_tile_wmma_f16(
    const _Float16* __restrict__ p0,
    long long p0_s0,
    const _Float16* __restrict__ p1,
    long long p1_s0,
    float* __restrict__ p2,
    long long p2_s0
) {
    const int rf_lane = (int)__lane_id();
    rf_v16f16 t0 = {};
    rf_v16f16 t1 = {};
    rf_v8f32 t2 = {};
    {
        const long long rf_epr = (long long)(((int)16LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        t0[0] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 0) * p0_s0];
        t0[1] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 1) * p0_s0];
        t0[2] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 2) * p0_s0];
        t0[3] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 3) * p0_s0];
        t0[4] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 4) * p0_s0];
        t0[5] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 5) * p0_s0];
        t0[6] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 6) * p0_s0];
        t0[7] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 7) * p0_s0];
        t0[8] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 8) * p0_s0];
        t0[9] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 9) * p0_s0];
        t0[10] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 10) * p0_s0];
        t0[11] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 11) * p0_s0];
        t0[12] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 12) * p0_s0];
        t0[13] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 13) * p0_s0];
        t0[14] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 14) * p0_s0];
        t0[15] = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + 15) * p0_s0];
    }
    {
        const long long rf_epr = (long long)(((int)16LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        t1[0] = p1[((long long)(((int)0LL)) + (rf_o0 + 0) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[1] = p1[((long long)(((int)0LL)) + (rf_o0 + 1) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[2] = p1[((long long)(((int)0LL)) + (rf_o0 + 2) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[3] = p1[((long long)(((int)0LL)) + (rf_o0 + 3) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[4] = p1[((long long)(((int)0LL)) + (rf_o0 + 4) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[5] = p1[((long long)(((int)0LL)) + (rf_o0 + 5) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[6] = p1[((long long)(((int)0LL)) + (rf_o0 + 6) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[7] = p1[((long long)(((int)0LL)) + (rf_o0 + 7) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[8] = p1[((long long)(((int)0LL)) + (rf_o0 + 8) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[9] = p1[((long long)(((int)0LL)) + (rf_o0 + 9) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[10] = p1[((long long)(((int)0LL)) + (rf_o0 + 10) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[11] = p1[((long long)(((int)0LL)) + (rf_o0 + 11) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[12] = p1[((long long)(((int)0LL)) + (rf_o0 + 12) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[13] = p1[((long long)(((int)0LL)) + (rf_o0 + 13) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[14] = p1[((long long)(((int)0LL)) + (rf_o0 + 14) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
        t1[15] = p1[((long long)(((int)0LL)) + (rf_o0 + 15) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
    }
    {
        const float rf_v = 0.0f;
        t2[0] = rf_v;
        t2[1] = rf_v;
        t2[2] = rf_v;
        t2[3] = rf_v;
        t2[4] = rf_v;
        t2[5] = rf_v;
        t2[6] = rf_v;
        t2[7] = rf_v;
    }
    t2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(t0, t1, t2);
    {
        const long long rf_epr = (long long)(((int)16LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        p2[((long long)(((int)0LL)) + (rf_o0 + (2 * 0 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[0];
        p2[((long long)(((int)0LL)) + (rf_o0 + (2 * 1 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[1];
        p2[((long long)(((int)0LL)) + (rf_o0 + (2 * 2 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[2];
        p2[((long long)(((int)0LL)) + (rf_o0 + (2 * 3 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[3];
        p2[((long long)(((int)0LL)) + (rf_o0 + (2 * 4 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[4];
        p2[((long long)(((int)0LL)) + (rf_o0 + (2 * 5 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[5];
        p2[((long long)(((int)0LL)) + (rf_o0 + (2 * 6 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[6];
        p2[((long long)(((int)0LL)) + (rf_o0 + (2 * 7 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[7];
    }
}
