#include "variables.h"
#include "funs.h"

void init_fill(float * mx, float * my, float * mz, const grid_color color,
               const int L, trng::lcg64_shift * Rng, trng::uniform_dist<float> Uni){

    float tmp1 = 0.0f, tmp2 = 0.0f, r0 = 0.0f;
    int x = 0, y = 0, z = 0;

    int ixf = 0;      // i on full size matrix
    int index = 0;    // index for full
    int rbi = 0;      // index for color matrix

    float res = 0.0f;
    int t_id;
    #pragma omp parallel for collapse(3) default(none) shared(mx, my, mz, L, Rng, Uni, color)\
    private(rbi, index, ixf, tmp1, tmp2, res, t_id)
    for (z=0; z<2*L; z++){
        for (y=0; y<2*L; y++){
            for (x=0; x<L; x++){
                t_id = omp_get_thread_num();
                ixf = 2*x + STRF(color, y, z);
                index = IX(ixf, y, z);
                rbi = RB(x, y, z);

                res = myrand(Rng, Uni, t_id);

                tmp1 = acos(2.0*(0.5-res));
                tmp2 = dospi*myrand(Rng, Uni, t_id);
                mx[rbi] = sin(tmp1)*cos(tmp2);
                my[rbi] = sin(tmp1)*sin(tmp2);
                mz[rbi] = cos(tmp1);
            }
        }
    }
}

//      Here we set the interaction constants matrix for one color
void init_oc(int * oc, float * macr, float * mafe, const int L,
             trng::lcg64_shift * Rng, trng::uniform_dist<float> Uni,
             grid_color color){

    int x = 0, y = 0, z = 0;
    float tmp3 = 0.0f;
    int index = 0;
    int rbi = 0;
    float res = 0.0f;
    int t_id = 0;
    int ixf = 0;
    #pragma omp parallel for collapse(3) default(none) shared(oc, macr, mafe, L, Rng, Uni, frac, color)\
    private(index, tmp3, res, t_id, rbi, ixf)
    for (z=0; z<2*L; z++){
        for (y=0; y<2*L; y++){
            for (x=0; x<L; x++){
                ixf = 2*x + STRF(color, y, z);
                index = IX(ixf, y, z);
                rbi = RB(x, y, z);
                t_id = omp_get_thread_num();
                res = myrand(Rng, Uni,t_id);

                if (res<= frac) {
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
    }
}

void interac_init(int * oc, float * KA,
               float * Jx, float * Jy, float * Jz,
               float * Dx, float * Dy, float * Dz,
               const grid_color color, const int L){

    int x = 0, y = 0, z = 0;
    int index = 0, rbi = 0, ixf = 0;
    int idx_i1 = 0, idx_j1 = 0, idx_k1 = 0;

    #pragma omp parallel for collapse(3) default(none) shared(oc, KA, Jx, Jy, Jz, Dx, Dy, Dz, L, color)\
     private(index, rbi, ixf, idx_i1, idx_j1, idx_k1)
    for (z=0; z<2*L; z++){
        for (y=0; y<2*L; y++){
            for (x=0; x<L; x++){

                ixf = 2*x + STRF(color, y, z);
                index = IX(ixf, y, z);
                rbi = RB(x, y, z);

                /* since Y and Z are the same for color and full matrixes, only ix is
                needed for consulting neighbors */
                idx_i1 = IX(c(ixf+1, 2*L), y, z);
                idx_j1 = IX(ixf, c(y+1, 2*L), z);
                idx_k1 = IX(ixf, y, c(z+1, 2*L));

                if (oc[index] == 1) {
                    // Defines anisotropic value per site
                    KA[rbi] = Kcr;
                    if (oc[idx_i1] == 1) {
                        Jx[rbi] = Jcrcr;
                        Dx[rbi] = Dcrcr;
                    }
                    else {
                        Jx[rbi] = Jfecr;
                        Dx[rbi] = Dfecr;
                    }

                    if (oc[idx_j1] == 1) {
                        Jy[rbi] = Jcrcr;
                        Dy[rbi] = Dcrcr;
                    }
                    else {
                        Jy[rbi] = Jfecr;
                        Dy[rbi] = Dfecr;
                    }

                    if (oc[idx_k1] == 1) {
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
                    if (oc[idx_i1] == 2) {
                        Jx[rbi] = Jfefe;
                        Dx[rbi] = Dfefe;
                    }
                    else {
                        Jx[rbi] = Jfecr;
                        Dx[rbi] = Dfecr;
                    }

                    if (oc[idx_j1] == 2) {
                        Jy[rbi] = Jfefe;
                        Dy[rbi] = Dfefe;
                    }
                    else {
                        Jy[rbi] = Jfecr;
                        Dy[rbi] = Dfecr;
                    }

                    if (oc[idx_k1] == 2) {
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
    }

}

void mc_init(int * oc, float * mc, const int L, grid_color color){

    int x = 0, y = 0, z = 0;
    int index = 0, ixf = 0, rbi = 0;

    #pragma omp parallel for collapse(3) default(none) shared(mc, oc, color, L) private(index, ixf, rbi)
    for (z=0; z<2*L; z++){
        for (y=0; y<2*L; y++){
            for (x=0; x<L; x++){
                ixf = 2*x + STRF(color, y, z);
                index = IX(ixf, y, z);
                rbi = RB(x, y, z);
                if (oc[index] == 2){
                    mc[rbi] = mfe;
                } else {
                    mc[rbi] = mcr;
                }
            }
        }
    }
}