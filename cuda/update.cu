#include <iostream>

#include <Random123/philox.h>
#include <Random123/u01fixedpt.h>

#include "helper_cuda.h"
#include "vector_types.h"
#include "cub/cub.cuh"

#include "variables.h"
#include "update.h"

typedef r123::Philox4x32 RNG4;

const static float pi_d = 3.14159265358979323846264338328f;
const static float dospi_d = 2.0f*pi_d;

__device__
void rand_quat(const unsigned int index, const unsigned int t, float * r1,
               float * r2, float * r3, float * r4) {

    RNG4 rng;
    RNG4::ctr_type c_pair = { {} };
    RNG4::key_type k_pair = { {} };
    RNG4::ctr_type r_quartet;
    // keys
    k_pair[0] = t+index;//philox_seed
    // time counter
    c_pair[0] = t;
    c_pair[1] = index;
    // random number generation
    r_quartet = rng(c_pair, k_pair);

    *r1 = u01fixedpt_open_open_32_float(r_quartet[0]);
    *r2 = u01fixedpt_open_open_32_float(r_quartet[1]);
    *r3 = u01fixedpt_open_open_32_float(r_quartet[2]);
    *r4 = u01fixedpt_open_open_32_float(r_quartet[3]);

}


__global__
void up_1(const float * __restrict__ mx, const float * __restrict__ my, const float * __restrict__ mz,
          const float * const __restrict__ KA, const float * const __restrict__ mc,
          const float * const __restrict__ Jx_s, const float * const __restrict__ Jy_s,
          const float * const __restrict__ Jz_s,
          const float * const __restrict__ Dx_s, const float * const __restrict__ Dy_s,
          const float * const __restrict__ Dz_s,
          const float * const __restrict__ Jx_o, const float * const __restrict__ Jy_o,
          const float * const __restrict__ Jz_o,
          const float * const __restrict__ Dx_o, const float * const __restrict__ Dy_o,
          const float * const __restrict__ Dz_o,
          const int L, const float Hx, const float Hy, const float Hz,
          const float Temp, const unsigned int time, int * __restrict__ in,
          const grid_color color,
          float * __restrict__ new_x, float * __restrict__ new_y, float * __restrict__ new_z,
          const float * const __restrict__ other_x, const float * const __restrict__ other_y,
          const float * const __restrict__ other_z) {

    int x, y, z;
    x = threadIdx.x + blockDim.x*blockIdx.x;
    y = threadIdx.y + blockDim.y*blockIdx.y;
    z = threadIdx.z + blockDim.z*blockIdx.z;

    if (x < L && y < 2*L && z < 2*L) {

        int index = 0, idx_rb = 0, ixf = 0;
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        float sen_theta = 0.0f, theta_n = 0.0f, phi_n = 0.0f;
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        float mxn = 0.0f, myn = 0.0f, mzn = 0.0f;
        float E1 = 0.0f, E2 = 0.0f, E3 = 0.0f;
        float Delta_E = 0.0f;

        int rbxp = 0, rbyp = 0, rbzp = 0;
        int rbxm = 0, rbym = 0, rbzm = 0;

        int counter = 0;
        int n = 2*L;

        float delta_x = 0.0f;
        float delta_y = 0.0f;
        float delta_z = 0.0f;

        float xxm = 0.0f, zxm=0.0f;
        float xym = 0.0f, zym=0.0f;
        float xzm = 0.0f, zzm=0.0f;

        float xxp = 0.0f, zxp=0.0f;
        float xyp = 0.0f, zyp=0.0f;
        float xzp = 0.0f, zzp=0.0f;

        counter = 0;
        ixf = 2*x + STRF(color, y, z);   // x on full size matrix
        index = IX(ixf, y, z);           // index for full size
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

        // old spin coordinates
        px = mx[idx_rb];
        py = my[idx_rb];
        pz = mz[idx_rb];

        rand_quat(index, time, &r0, &r1, &r2, &r3);

        theta_n = acosf(2.0f*(0.5f - r0));
        phi_n = dospi_d*r1;
        sen_theta = sinf(theta_n);

        // new candidates to spin coordinates
        mxn = sen_theta*cosf(phi_n);
        myn = sen_theta*sinf(phi_n);
        mzn = cosf(theta_n);

        delta_x = mxn - px;
        delta_y = myn - py;
        delta_z = mzn - pz;

        // iones have different magnetic moment hence we use mc[x,y,z]
        E1 = -KA[idx_rb]*(mxn*mxn - px*px) - (Hz*(delta_z) + Hy*(delta_y) + Hx*(delta_x))*mc[idx_rb];

        // this reads are saved becaause they showed an improvement in performance
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

        // E2 - exchange term: the spin interacts with its 6 neighbours
        // E3 - DM Interaction: the spin vectorialy interacts
        // E3 - Interaccion DM: el espin interactua vectorialente con sus vecinos
        E2 += (xxm*(delta_x) + other_y[rbxm]*(delta_y) + zxm*(delta_z))*Jx_o[rbxm];
        E3 += (zxm*(delta_x) - xxm*(delta_z))*Dx_o[rbxm];

        E2 += (xym*(delta_x) + other_y[rbym]*(delta_y) + zym*(delta_z))*Jy_o[rbym];
        E3 += (zym*(delta_x) - xym*(delta_z))*Dy_o[rbym];

        E2 += (xzm*(delta_x) + other_y[rbzm]*(delta_y) + zzm*(delta_z))*Jz_o[rbzm];
        E3 += (zzm*(delta_x) - xzm*(delta_z))*Dz_o[rbzm];

        E2 += (xxp*(delta_x) + other_y[rbxp]*(delta_y) + zxp*(delta_z))*Jx_s[idx_rb];
        E3 += (zxp*(delta_x) - xxp*(delta_z))*Dx_s[idx_rb];

        E2 += (xyp*(delta_x) + other_y[rbyp]*(delta_y) + zyp*(delta_z))*Jy_s[idx_rb];
        E3 += (zyp*(delta_x) - xyp*(delta_z))*Dy_s[idx_rb];

        E2 += (xzp*(delta_x) + other_y[rbzp]*(delta_y) + zzp*(delta_z))*Jz_s[idx_rb];
        E3 += (zzp*(delta_x) - xzp*(delta_z))*Dz_s[idx_rb];

        // this flip depends on the coordinates of the full size matrix
        if (!((ixf+y+z)&1)) {
            E3 = -E3;
        }

        // Metropolis Algorithm
        Delta_E = E1 + E2 + E3;

        if (Delta_E < 0.0f) {
            // if the energy lowers the change is accepted
            new_x[idx_rb] = mxn;
            new_y[idx_rb] = myn;
            new_z[idx_rb] = mzn;
            counter = 1;
        }
        else if (r3 < expf(-Delta_E/Temp)) {
            // if it goes up it gets a prob
            new_x[idx_rb] = mxn;
            new_y[idx_rb] = myn;
            new_z[idx_rb] = mzn;
            counter = 1;
        } else {
            // else keep old values.
            new_x[idx_rb] = px;
            new_y[idx_rb] = py;
            new_z[idx_rb] = pz;
        }

        in[idx_rb] = counter;
    }

}

__host__
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
             float * __restrict__ black_x, float * __restrict__ black_y, float * __restrict__ black_z) {

    up_1<<<blocks, threads>>>(mx_r, my_r, mz_r, KA_r, mc_r, Jx_r, Jy_r, Jz_r,
            Dx_r, Dy_r, Dz_r, Jx_b, Jy_b, Jz_b, Dx_b, Dy_b, Dz_b, L, Hx, Hy,
            Hz, Temp, time, in1, RED_TILES, red_x, red_y, red_z, mx_b,
            my_b, mz_b);
    getLastCudaError("update RED failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    up_1<<<blocks, threads>>>(mx_b, my_b, mz_b, KA_b, mc_b, Jx_b, Jy_b, Jz_b,
            Dx_b, Dy_b, Dz_b, Jx_r, Jy_r, Jz_r, Dx_r, Dy_r, Dz_r, L, Hx, Hy,
            Hz, Temp, time, in2, BLACK_TILES, black_x, black_y, black_z,
            red_x, red_y, red_z);
    getLastCudaError("update BLACK failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

}
