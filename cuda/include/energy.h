#ifndef ENERGY_H_INCLUDED
#define ENERGY_H_INCLUDED

void energy(const float * const __restrict__ mx_r, const float * const __restrict__ my_r,
            const float * const __restrict__ mz_r,
            const float * const __restrict__ mx_b, const float * const __restrict__ my_b,
            const float * const __restrict__ mz_b,
            float * __restrict__ pE_afm_ex, float * __restrict__ pE_afm_an, float * __restrict__ pE_afm_dm,
            const float * const __restrict__ KA_r, const float * const __restrict__ KA_b,
            const float * const __restrict__ mc_r, const float * const __restrict__ mc_b,
            const float * const __restrict__ Jx_r, const float * const __restrict__ Jy_r,
            const float * const __restrict__ Jz_r,
            const float * const __restrict__ Jx_b, const float * const __restrict__ Jy_b,
            const float * const __restrict__ Jz_b,
            const float * const __restrict__ Dx_r, const float * const __restrict__ Dy_r,
            const float * const __restrict__ Dz_r,
            const float * const __restrict__ Dx_b, const float * const __restrict__ Dy_b,
            const float * const __restrict__ Dz_b,
            const float Hx, const float Hy, const float Hz,
            float* am1, float* am2, float* am3, float* am4, float* am5, float* am6, float* aux_mat,
            void * d_ts, size_t d_ts_bytes, const int L,
            const dim3 blocks, const dim3 threads);

#endif
