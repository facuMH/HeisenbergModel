#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <math.h>

#include "helper_cuda.h"
#include "cub/cub.cuh"

#include "variables.h"
#include "energy.h"

__global__
void energy_kernel(const float * const __restrict__ mx, const float * const __restrict__ my,
                   const float * const __restrict__ mz,
                   float * __restrict__ pE_afm_ex, float * __restrict__ pE_afm_an,
                   float * __restrict__ pE_afm_dm, const float * const __restrict__ KA,
                   const float * const __restrict__ mc,
                   const float * const __restrict__ Jx_s, const float * const __restrict__ Jy_s,
                   const float * const __restrict__ Jz_s,
                   const float * const __restrict__ Jx_o, const float * const __restrict__ Jy_o,
                   const float * const __restrict__ Jz_o,
                   const float * const __restrict__ Dx_s, const float * const __restrict__ Dy_s,
                   const float * const __restrict__ Dz_s,
                   const float * const __restrict__ Dx_o, const float * const __restrict__ Dy_o,
                   const float * const __restrict__ Dz_o,
                   const float Hx, const float Hy, const float Hz,
                   float * __restrict__ am1, float * __restrict__ am2, float * __restrict__ am3,
                   const int L, const grid_color color,
                   const float * const __restrict__ other_x, const float * const __restrict__ other_y,
                   const float * const __restrict__ other_z){

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;

    if (x < L && y < 2*L && z < 2*L) {

        const int n = L*2;
        int index = 0, ixf = 0, idx_rb = 0;

        float px = 0.0f;
        float py = 0.0f;
        float pz = 0.0f;
        float ener = 0.0f;

        float E_afm_ex = 0.0f;
        float E_afm_an = 0.0f;
        float E_afm_dm = 0.0f;

        float xxm = 0.0f, zxm=0.0f;
        float xym = 0.0f, zym=0.0f;
        float xzm = 0.0f, zzm=0.0f;

        float xxp = 0.0f, zxp=0.0f;
        float xyp = 0.0f, zyp=0.0f;
        float xzp = 0.0f, zzp=0.0f;

        int rbzm = 0;
        int rbym = 0;
        int rbxm = 0;

        int rbzp = 0;
        int rbyp = 0;
        int rbxp = 0;

        ixf = 2*x + STRF(color, y, z);   // i on full size matrix
        index = IX(ixf, y, z);
        idx_rb = RB(x, y, z);            // index for color matrix

        /* for the neightbours in the matrix of the other colors,
         * the ones in X are same index and one more or one less,
         * depending on index parity.
         */
        if (!(index&1)){
            rbxp = RB(x, y, z);
            rbxm = RB(c(x - 1, L), y, z);
        } else {
            rbxp = RB(c(x + 1, L), y, z);
            rbxm = RB(x, y, z);
        }

        rbyp = RB(x, c(y + 1, n), z);
        rbzp = RB(x, y, c(z + 1, n));

        rbym = RB(x, c(y - 1, n), z);
        rbzm = RB(x, y, c(z - 1, n));

        px = mx[idx_rb];
        py = my[idx_rb];
        pz = mz[idx_rb];

        // anisortopy energy
        E_afm_an = -KA[idx_rb]*(px*px) - (Hx*px + Hy*py + Hz*pz)*mc[idx_rb];
        am1[idx_rb] = E_afm_an;

        xxm = other_x[rbxm];
        zxm = other_z[rbxm];
        xym = other_x[rbym];
        zym = other_z[rbym];
        xzm = other_x[rbzm];
        zzm = other_z[rbzm];

        xxp = other_x[rbxp];
        zxp = other_z[rbxp];
        xyp = other_x[rbyp];
        zyp = other_z[rbyp];
        xzp = other_x[rbzp];
        zzp = other_z[rbzp];

        // exchange energy
        E_afm_ex += (xxm*px + other_y[rbxm]*py + zxm*pz)*Jx_o[rbxm];
        ener += (zxm*px - xxm*pz)*Dx_o[rbxm];

        E_afm_ex += (xym*px + other_y[rbym]*py + zym*pz)*Jy_o[rbym];
        ener += (zym*px - xym*pz)*Dy_o[rbym];

        E_afm_ex += (xzm*px + other_y[rbzm]*py + zzm*pz)*Jz_o[rbzm];
        ener += (zzm*px - xzm*pz)*Dz_o[rbzm];

        E_afm_ex += (xxp*px + other_y[rbxp]*py + zxp*pz)*Jx_s[idx_rb];
        ener += (zxp*px - xxp*pz)*Dx_s[idx_rb];

        E_afm_ex += (xyp*px + other_y[rbyp]*py + zyp*pz)*Jy_s[idx_rb];
        ener += (zyp*px - xyp*pz)*Dy_s[idx_rb];

        E_afm_ex += (xzp*px + other_y[rbzp]*py + zzp*pz)*Jz_s[idx_rb];
        ener += (zzp*px - xzp*pz)*Dz_s[idx_rb];

        am2[idx_rb] = E_afm_ex;

        E_afm_dm = -(ener*(1 - 2*((ixf+y+z)&1)));
        am3[idx_rb] = E_afm_dm;
    }
}

__host__
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
            const dim3 blocks, const dim3 threads) {

    float E_afm_an = 0.0f;
    float E_afm_ex = 0.0f;
    float E_afm_dm = 0.0f;
    const int RBN = 4*L*L*L;

    energy_kernel<<<blocks, threads>>>(mx_r, my_r, mz_r, &E_afm_ex, &E_afm_an,
                        &E_afm_dm, KA_r, mc_r, Jx_r, Jy_r, Jz_r, Jx_b, Jy_b, Jz_b,
                        Dx_r, Dy_r, Dz_r, Dx_b, Dy_b, Dz_b, Hx, Hy, Hz,
                        am1, am2, am3, L, RED_TILES, mx_b, my_b, mz_b);
    getLastCudaError("energy red failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am1, aux_mat, RBN));
    getLastCudaError("reduct red energy 1 failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    E_afm_an = aux_mat[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am2, aux_mat, RBN));
    getLastCudaError("reduct red energy 2 failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    E_afm_ex = aux_mat[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am3, aux_mat, RBN));
    getLastCudaError("reduct red energy 3 failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    E_afm_dm = aux_mat[0];

    energy_kernel<<<blocks, threads>>>(mx_b, my_b, mz_b, &E_afm_ex, &E_afm_an,
                        &E_afm_dm, KA_b, mc_b, Jx_b, Jy_b, Jz_b, Jx_r, Jy_r, Jz_r,
                        Dx_b, Dy_b, Dz_b, Dx_r, Dy_r, Dz_r, Hx, Hy, Hz,
                        am4, am5, am6, L, BLACK_TILES, mx_r, my_r, mz_r);
    getLastCudaError("energy black failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am4, aux_mat, RBN));
    getLastCudaError("reduct black energy 1 failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    E_afm_an += aux_mat[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am5, aux_mat, RBN));
    getLastCudaError("reduct black energy 2 failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    E_afm_ex += aux_mat[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am6, aux_mat, RBN));
    getLastCudaError("reduct black energy 3 failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    E_afm_dm += aux_mat[0];

    *pE_afm_ex = 0.5f*E_afm_ex/(RBN*2);  //la energia de intercambio se cuenta dos veces
    *pE_afm_dm = 0.5f*E_afm_dm/(RBN*2);  //la energia de intercambio se cuenta dos veces
    *pE_afm_an = E_afm_an/(RBN*2);
}
