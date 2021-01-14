#include <cstdlib>
#include <iostream>

#include "variables.h"
#include "update.h"

void update(const float * const mx, const float * const my, const float * const mz,
           const float * const KA, const float * const mc, const float * const Jx_s,
           const float * const Jy_s, const float * const Jz_s,
           const float * const Dx_s, const float * const Dy_s,
           const float * const Dz_s, const float * const Jx_o,
           const float * const Jy_o, const float * const Jz_o,
           const float * const Dx_o, const float * const Dy_o,
           const float * const Dz_o, const grid_color color,
           float * new_x, float * new_y, float * new_z,
           const float * const other_x, const float * const other_y,
           const float * const other_z, const int L, const float Hx,
           const float Hy, const float Hz, const float Temp,
           int * cont, trng::lcg64_shift * Rng, trng::uniform_dist<float> Uni){

    int x = 0, y = 0, z = 0;
    float sen_theta = 0.0f, theta_n = 0.0f, phi_n = 0.0f;
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float mxn = 0.0f, myn = 0.0f, mzn = 0.0f;
    float E1 = 0.0f, E2 = 0.0f, E3 = 0.0f;
    float Delta_E = 0.0f;

    int changes = 0;

    int ixf = 0, index = 0, idx_rb = 0;
    int cxm = 0, cxp = 0; // c( x minus 1) -- c( x plus 1)
    int cym = 0, cyp = 0;
    int czm = 0, czp = 0;

    int rbxp = 0, rbyp = 0, rbzp = 0;
    int rbxm = 0, rbym = 0, rbzm = 0;

    const int n = 2*L;

    bool first = false, second = false;
    float exponential = 0.0f;

    float res = 0.0f;

    int t_id = 0;
    #pragma omp parallel for collapse(2) default(none) shared(new_x, new_y,\
    new_z, cont, dirty, std::cout, Rng, Uni) private(cxm, cxp, cym, cyp,\
    czm, czp, index, px, py, pz, mxn, myn, mzn, E1, E2, E3, theta_n, phi_n,\
    sen_theta, Delta_E, rbxp, rbyp, rbzp, rbxm, rbym, rbzm, ixf, idx_rb,\
    first, second, exponential, t_id, res, x)

    for (z=0; z<2*L; z++){
        for (y=0; y<2*L; y++){
            t_id = omp_get_thread_num();
            for (x=0; x<L; x++){

                ixf = 2*x + STRF(color, y, z);   // i on full size matrix
                index = IX(ixf, y, z);           // index for full
                idx_rb = RB(x, y, z);            // index for color matrix

                /* for the neightbours in the matrix of the other colors,
                * the ones in X are same index and one more or one less,
                * depending on index parity.
                */
                if (!(index&1)){
                    cxm = c(x - 1, L); // index even
                    cxp = x;
                } else {
                    cxm = x;
                    cxp = c(x + 1, L); // index odd
                }

                cym = c(y-1,n);
                cyp = c(y+1,n);
                czm = c(z-1,n);
                czp = c(z+1,n);

                rbxp = RB(cxp, y, z);
                rbyp = RB(x, cyp, z);
                rbzp = RB(x, y, czp);
                rbxm = RB(cxm, y, z);
                rbym = RB(x, cym, z);
                rbzm = RB(x, y, czm);

                 // old spin coordinates
                px = mx[idx_rb];
                py = my[idx_rb];
                pz = mz[idx_rb];

                res = myrand(Rng, Uni, t_id);

                theta_n = acosf(2.0f*(0.5f-res));
                phi_n = dospi*myrand(Rng, Uni, t_id);
                sen_theta = sinf(theta_n);

                // new candidates to spin coordinates
                mxn = sen_theta*cosf(phi_n);
                myn = sen_theta*sinf(phi_n);
                mzn = cosf(theta_n);

                // iones have different magnetic moment hence we use mc[x,y,z]
                E1 = -KA[idx_rb]*((mxn*mxn) - (px*px)) - (Hz*(mzn - pz) + Hy*(myn - py) + Hx*(mxn - px))*mc[idx_rb];

                E2 = 0.0f;
                E3 = 0.0f;

                // E2 - exchange term: the spin interacts with its 6 neighbours
                // E3 - DM Interaction: the spin vectorialy interacts
                E2 += (other_x[rbxm]*(mxn - px) + other_y[rbxm]*(myn - py) + other_z[rbxm]*(mzn - pz))*Jx_o[rbxm];
                E3 += (other_z[rbxm]*(mxn - px) - other_x[rbxm]*(mzn - pz))*Dx_o[rbxm];
                E2 += (other_x[rbym]*(mxn - px) + other_y[rbym]*(myn - py) + other_z[rbym]*(mzn - pz))*Jy_o[rbym];
                E3 += (other_z[rbym]*(mxn - px) - other_x[rbym]*(mzn - pz))*Dy_o[rbym];
                E2 += (other_x[rbzm]*(mxn - px) + other_y[rbzm]*(myn - py) + other_z[rbzm]*(mzn - pz))*Jz_o[rbzm];
                E3 += (other_z[rbzm]*(mxn - px) - other_x[rbzm]*(mzn - pz))*Dz_o[rbzm];

                E2 += (other_x[rbxp]*(mxn - px) + other_y[rbxp]*(myn - py) + other_z[rbxp]*(mzn - pz))*Jx_s[idx_rb];
                E3 += (other_z[rbxp]*(mxn - px) - other_x[rbxp]*(mzn - pz))*Dx_s[idx_rb];
                E2 += (other_x[rbyp]*(mxn - px) + other_y[rbyp]*(myn - py) + other_z[rbyp]*(mzn - pz))*Jy_s[idx_rb];
                E3 += (other_z[rbyp]*(mxn - px) - other_x[rbyp]*(mzn - pz))*Dy_s[idx_rb];
                E2 += (other_x[rbzp]*(mxn - px) + other_y[rbzp]*(myn - py) + other_z[rbzp]*(mzn - pz))*Jz_s[idx_rb];
                E3 += (other_z[rbzp]*(mxn - px) - other_x[rbzp]*(mzn - pz))*Dz_s[idx_rb];

                // this flip depends on the coordinates of the full size matrix
                if (!((ixf+y+z)&1)) {
                    E3 = -E3;
                }

                // Metropolis Algorithm
                Delta_E = E1 + E2 + E3;

                first = Delta_E < 0.0f;
                exponential = expf(-Delta_E/Temp);
                second = myrand(Rng, Uni, t_id) < exponential;

                if (first ) {
                    // if the energy lowers the change is accepted
                    new_x[idx_rb] = mxn;
                    new_y[idx_rb] = myn;
                    new_z[idx_rb] = mzn;
                    cont[idx_rb] = 1;
                }
                else if (second) {
                    // if it goes up it gets a prob
                    new_x[idx_rb] = mxn;
                    new_y[idx_rb] = myn;
                    new_z[idx_rb] = mzn;
                    cont[idx_rb] = 1;
                } else {
                    // else keep old values.
                    new_x[idx_rb] = mx[idx_rb];
                    new_y[idx_rb] = my[idx_rb];
                    new_z[idx_rb] = mz[idx_rb];
                }

            }
        }
    }

    return;
}
