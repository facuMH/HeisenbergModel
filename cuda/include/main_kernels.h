

__host__
void calculate(float * __restrict__ var1_x, float * __restrict__ var1_y, float * __restrict__ var1_z,
               float * __restrict__ var2_x, float * __restrict__ var2_y, float * __restrict__ var2_z,
               const float * const __restrict__ mx_r, const float * const __restrict__ my_r,
               const float * const __restrict__ mz_r,
               const float * const __restrict__ mx_b, const float * const __restrict__ my_b,
               const float * const __restrict__ mz_b,
               const float * const __restrict__ elem_r, const float * const __restrict__ elem_b,
               float * __restrict__ am1_x, float * __restrict__ am1_y, float * __restrict__ am1_z,
               float * __restrict__ am2_x, float * __restrict__ am2_y, float * __restrict__ am2_z,
               float * __restrict__ out, void * __restrict__ d_ts, const __restrict__ size_t d_ts_bytes,
               const int L, const dim3 blocks, const dim3 threads);


