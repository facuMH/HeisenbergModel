#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <math.h>

#include "variables.h"
#include "energy.h"

void energy(const float * const mx, const float * const my,
            const float * const mz, float * E_afm_ex, float * E_afm_an,
            float * E_afm_dm, const float * const KA, const float * const mc,
            const float * const Jz_s, const float * const Jy_s,
            const float * const Jx_s, const float * const Dz_s,
            const float * const Dy_s, const float * const Dx_s,
            const float * const Jz_o, const float * const Jy_o,
            const float * const Jx_o, const float * const Dz_o,
            const float * const Dy_o, const float * const Dx_o,
            const float * const other_x, const float * const other_y,
            const float * const other_z, const grid_color color, const int L){

    int x = 0, y = 0, z = 0;
    float px = 0.0f, py = 0.0f, pz = 0.0f, ener = 0.0f;

    const int n = 2*L;
    
    int index = 0, idx_rb = 0, ixf = 0;
    int cxm = 0, cxp = 0; // circular(x minus 1) -- circular(x plus 1)
    int cym = 0, cyp = 0; // circular(y minus 1) -- circular(y plus 1)
    int czm = 0, czp = 0; // circular(z minus 1) -- circular(z plus 1)

    int idx_zm = 0;
    int idx_ym = 0;
    int idx_xm = 0;

    int idx_zp = 0;
    int idx_yp = 0;
    int idx_xp = 0;

    float local_ex = 0.0f, local_an  = 0.0f, local_dm = 0.0f;

    #pragma omp parallel for collapse(2) shared(E_afm_ex,E_afm_an, E_afm_dm) \
    private(cxm, cxp, cym, cyp, czm, czp,idx_zm,idx_ym,idx_xm,idx_zp,\
    idx_yp, idx_xp,index, px, py, pz, ener, local_ex, local_an, local_dm, x)
    for (z=0; z<2*L; z++){
        for (y=0; y<2*L; y++){
            local_ex = 0.0f;
            local_an = 0.0f;
            local_dm = 0.0f;
            for (x=0; x<L; x++){

                ixf = 2*x + STRF(color,y, z);   // i on full size matrix
                idx_rb = RB(x, y, z);        // index for color matrix

                px = mx[idx_rb];
                py = my[idx_rb];
                pz = mz[idx_rb];

                cxm = c(x - 1, L);
                cxp = c(x + 1, L);
                cym = c(y - 1, n);
                cyp = c(y + 1, n);
                czm = c(z - 1, n);
                czp = c(z + 1, n);

                idx_zm = IX(x, y, czm);
                idx_ym = IX(x, cym, z);
                idx_xm = IX(cxm, y, z);
                idx_zp = IX(x, y, czp);
                idx_yp = IX(x, cyp, z);
                idx_xp = IX(cxp, y, z);

                ener = 0.0f;
                // Energia de anisotropia
                local_an -= KA[idx_rb]*(px*px) - (Hz*pz + Hy*py + Hx*px)*mc[idx_rb];

                // Energia de intercambio
                local_ex += (other_x[idx_xm]*px + other_y[idx_xm]*py + other_z[idx_xm]*pz)*Jx_o[idx_xm];
                ener += (other_z[idx_xm]*px - other_x[idx_xm]*pz)*Dx_o[idx_xm];

                local_ex += (other_x[idx_ym]*px + other_y[idx_ym]*py + other_z[idx_ym]*pz)*Jy_o[idx_ym];
                ener += (other_z[idx_ym]*px - other_x[idx_ym]*pz)*Dy_o[idx_ym];

                local_ex += (other_x[idx_zm]*px + other_y[idx_zm]*py + other_z[idx_zm]*pz)*Jz_o[idx_zm];
                ener += (other_z[idx_zm]*px - other_x[idx_zm]*pz)*Dz_o[idx_zm];

                local_ex += (other_x[idx_xp]*px + other_y[idx_xp]*py + other_z[idx_xp]*pz)*Jx_s[idx_rb];
                ener += (other_z[idx_xp]*px - other_x[idx_xp]*pz)*Dx_s[idx_rb];

                local_ex += (other_x[idx_yp]*px + other_y[idx_yp]*py + other_z[idx_yp]*pz)*Jy_s[idx_rb];
                ener += (other_z[idx_yp]*px - other_x[idx_yp]*pz)*Dy_s[idx_rb];

                local_ex += (other_x[idx_zp]*px + other_y[idx_zp]*py + other_z[idx_zp]*pz)*Jz_s[idx_rb];
                ener += (other_z[idx_zp]*px - other_x[idx_zp]*pz)*Dz_s[idx_rb];

                local_dm -= (ener*(1 - 2*((ixf+y+z)&1)));
            }

            E_afm_ex[y+2*L*z] = local_ex;
            E_afm_dm[y+2*L*z] = local_dm;
            E_afm_an[y+2*L*z] = local_an;
        }
    }
}
