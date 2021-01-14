
#ifndef UPDATE_H_INCLUDED
#define UPDATE_H_INCLUDED

void update (const float * const __restrict__ mx_r, const float * const __restrict__ my_r,
             const float * const __restrict__ mz_r,
             const float * const __restrict__ mx_b, const float * const __restrict__ my_b,
             const float * const __restrict__ mz_b,
             const float * const __restrict__ Jx_r, const float * const __restrict__ Jy_r,
             const float * const __restrict__ Jz_r,
             const float * const __restrict__ Dx_r, const float * const __restrict__ Dy_r,
             const float * const __restrict__ Dz_r,
             const float * const __restrict__ Jx_b, const float * const __restrict__ Jy_b,
             const float * const __restrict__ Jz_b,
             const float * const __restrict__ Dx_b, const float * const __restrict__ Dy_b,
             const float * const __restrict__ Dz_b,
             const int L, const float * const __restrict__ KA_r,  const float * const __restrict__ KA_b,
             const float * const __restrict__ mc_r, const float * const __restrict__ mc_b,
             const float Hx, const float Hy, const float Hz,
             const float Temp, const unsigned int time,
             int * __restrict__ in1, int * __restrict__ in2,
             const dim3 blocks, const dim3 threads,
             float * __restrict__ red_x, float * __restrict__ red_y, float * __restrict__ red_z,
             float * __restrict__ black_x, float * __restrict__ black_y, float * __restrict__ black_z);

#endif
