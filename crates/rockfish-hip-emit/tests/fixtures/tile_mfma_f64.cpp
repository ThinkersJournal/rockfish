#include <hip/hip_runtime.h>

typedef double rf_v4f64 __attribute__((ext_vector_type(4)));

extern "C" __global__ void rockfish_hip_k_tile_mfma_f64(
    const double* __restrict__ p0,
    long long p0_s0,
    const double* __restrict__ p1,
    long long p1_s0,
    double* __restrict__ p2,
    long long p2_s0
) {
    const int rf_lane = (int)__lane_id();
    double t0 = {};
    double t1 = {};
    rf_v4f64 t2 = {};
    {
        const long long rf_epr = (long long)(((int)4LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        t0 = p0[((long long)(((int)0LL)) + (rf_o0 + (rf_lane % 16)) * rf_epr + rf_o1 + (rf_lane / 16)) * p0_s0];
    }
    {
        const long long rf_epr = (long long)(((int)16LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        t1 = p1[((long long)(((int)0LL)) + (rf_o0 + (rf_lane / 16)) * rf_epr + rf_o1 + (rf_lane % 16)) * p1_s0];
    }
    {
        const double rf_v = 0.0;
        t2[0] = rf_v;
        t2[1] = rf_v;
        t2[2] = rf_v;
        t2[3] = rf_v;
    }
    t2 = __builtin_amdgcn_mfma_f64_16x16x4f64(t0, t1, t2, 0, 0, 0);
    {
        const long long rf_epr = (long long)(((int)16LL)), rf_o0 = (long long)(((int)0LL)), rf_o1 = (long long)(((int)0LL));
        p2[((long long)(((int)0LL)) + (rf_o0 + (4 * 0 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[0];
        p2[((long long)(((int)0LL)) + (rf_o0 + (4 * 1 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[1];
        p2[((long long)(((int)0LL)) + (rf_o0 + (4 * 2 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[2];
        p2[((long long)(((int)0LL)) + (rf_o0 + (4 * 3 + (rf_lane / 16))) * rf_epr + rf_o1 + (rf_lane % 16)) * p2_s0] = t2[3];
    }
}
