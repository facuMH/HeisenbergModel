#include "variables.h"
#include "funs.h"
#include <Random123/philox.h>
#include <Random123/u01fixedpt.h>

typedef r123::Philox2x32 RNG2;

__device__
void rand_pair(const unsigned int index, const unsigned int time,
               float * r0, float * r1) {

    RNG2 rng;
    RNG2::ctr_type c_pair = { {} };
    RNG2::key_type k_pair = { {} };
    RNG2::ctr_type r_pair;

    // keys
    k_pair[0] = time+index;
    // time counter
    c_pair[0] = time;
    c_pair[1] = index;
    // random number generation
    r_pair = rng(c_pair, k_pair);

    *r0 = u01fixedpt_open_open_32_double(r_pair[0]);
    *r1 = u01fixedpt_open_open_32_double(r_pair[1]);

}

__global__
void init_fill(float * mx, float * my, float * mz, const int L,
               const unsigned int time, const grid_color color) {

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;

    const float pi_d = 3.14159265358979323846264338328f;
    const float dospi_d = 2.0f*pi_d;
    float tmp1, tmp2, r0, r1;

    if (x < L && y < 2*L && z < 2*L) {

        int ixf = 2*x + STRF(color, y, z);   // i on full size matrix
        int index = IX(ixf, y, z);           // index for full
        int rbi = RB(x, y, z);               // index for color matrix

        rand_pair(index, time, &r0, &r1);
        tmp1 = acos(2.0f*(0.5f - r0));
        tmp2 = dospi_d*r1;
        mx[rbi] = sinf(tmp1)*cosf(tmp2);
        my[rbi] = sinf(tmp1)*sinf(tmp2);
        mz[rbi] = cosf(tmp1);
    }

}

__global__
void init_oc(int * oc, float * macr, float * mafe, const int L,
             const unsigned int time, const float frac,
             const float flucfrac, const grid_color color) {

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;

    float r0, r1;

    if (x < L && y < 2*L && z < 2*L) {

        int ixf = 2*x + STRF(color, y, z);   // x on full size matrix
        int index = IX(ixf, y, z);
        int rbi = RB(x, y, z);

        rand_pair(index, time, &r0, &r1);
        if (r0 <= frac) {
            oc[index] = 1;
            macr[rbi] = 1.0f;
            mafe[rbi] = 0.0f;
        } else {
            oc[index] = 2;
            mafe[rbi] = 1.0f;
            macr[rbi] = 0.0f;
        }

    }
}

//      Here we set the interaction constants matrix for one color
__global__
void interac_init(int * oc, float * KA, float * Jx, float * Jy, float * Jz,
               float * Dx, float * Dy, float * Dz, const int L,
               const grid_color color) {

    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    int k = threadIdx.z + blockDim.z*blockIdx.z;

    if (i < L && j < 2*L && k < 2*L) {

        int ixf = 2*i + STRF(color,j,k);   // i on full size matrix
        int index = IX(ixf, j, k);         // index for full
        int rbi = RB(i,j,k);               // index for color matrix
        /* since Y and Z are the same for color and full matrixes, only ix is
         needed for consulting neighbors */
        int idx_x1 = IX(c(ixf+1, 2*L), y, z);
        int idx_y1 = IX(ixf, c(y+1, 2*L), z);
        int idx_z1 = IX(ixf, y, c(z+1, 2*L));

        const float JJ = 1.0f;
        const float Jfefe = JJ;
        const float Jfecr = 153.0f/628.0f;
        const float Jcrcr = 115.0f/628.0f;
        const float Dfefe = 0.0214f*JJ;
        const float Dcrcr = 0.0074f*JJ;
        const float Dfecr = -0.0168f*JJ;

        const float Kcr = 0.007f * JJ;
        const float Kfe = 0.007f * JJ;

        if (oc[index] == 1) {
            // Defines anisotropic value per site
            KA[rbi] = Kcr;
            if (oc[idx_x1] == 1) {
                Jx[rbi] = Jcrcr;
                Dx[rbi] = Dcrcr;
            }
            else {
                Jx[rbi] = Jfecr;
                Dx[rbi] = Dfecr;
            }

            if (oc[idx_y1] == 1) {
                Jy[rbi] = Jcrcr;
                Dy[rbi] = Dcrcr;
            }
            else {
                Jy[rbi] = Jfecr;
                Dy[rbi] = Dfecr;
            }

            if (oc[idx_z1] == 1) {
                Jz[rbi] = Jcrcr;
                Dz[rbi] = Dcrcr;
            }
            else {
                Jz[rbi] = Jfecr;
                Dz[rbi] = Dfecr;
            }
        } else {
            //  Define el valor la anisotropia por sitio
            KA[rbi] = Kfe;
            if (oc[idx_x1] == 2) {
                Jx[rbi] = Jfefe;
                Dx[rbi] = Dfefe;
            }
            else {
                Jx[rbi] = Jfecr;
                Dx[rbi] = Dfecr;
            }

            if (oc[idx_y1] == 2) {
                Jy[rbi] = Jfefe;
                Dy[rbi] = Dfefe;
            }
            else {
                Jy[rbi] = Jfecr;
                Dy[rbi] = Dfecr;
            }

            if (oc[idx_z1] == 2) {
                Jz[rbi] = Jfefe;
                Dz[rbi] = Dfefe;
            }
            else {
                Jz[rbi] = Jfecr;
                Dz[rbi] = Dfecr;
            }
        }
    }

}


__global__
void mc_init(int * oc, float * mc, const int L, const grid_color color) {

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;

    if (x < L && y < 2*L && z < 2*L) {

        int ixf = 2*x + STRF(color, y, z);   // i on full size matrix
        int index = IX(ixf, y, z);
        int rbi = RB(x, y, z);

        const float mfe_d = 1.0f;                //-0.0168*JJ
        const float mcr_d = 0.6f;              //-0.0168*JJ
        if (oc[index] == 2) {
            mc[rbi] = mfe_d;
        }
        else {
            mc[rbi] = mcr_d;
        }
    }
}
