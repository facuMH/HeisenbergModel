#include "variables.h"
#include "main_kernels.h"

void calculate(const float * mx, const float * my,
               const float * mz, const float * const elem,
               float * var_x, float * var_y, float * var_z,
               const grid_color color, const int L){

    int index = 0, rbi = 0, ixf = 0, ix = 0;
    int x = 0, y = 0, z = 0;
    int L_4 = (L+4-1)/4;

    int rb_1 = 0, rb_2 = 0,rb_3 = 0;

    float local_x = 0.0f, local_y  = 0.0f, local_z = 0.0f;

    #pragma omp parallel for collapse(2) shared(var_x,var_y,var_z)\
    private(rb_1,rb_2,rb_3,index,rbi,ixf, local_x, local_y,\
    local_z, i)
    for (z=0; z<2*L; z++){
        for (y=0; y<2*L; y++){
            local_x = 0.0f;
            local_y = 0.0f;
            local_z = 0.0f;
            for (x=0; x<L; x++){

                ix = 4*x;
                rbi = RB(ix, y, z);        // index for color matrix

                // this indexs are the neighbours in the other color matrix
                rb_1 = RB(ix+1, y, z);
                rb_2 = RB(ix+2, y, z);
                rb_3 = RB(ix+3, y, z);

                local_x += mx[rbi]*elem[rbi];
                local_y += my[rbi]*elem[rbi];
                local_z += mz[rbi]*elem[rbi];

                local_x += mx[rb_1]*elem[rb_1];
                local_y += my[rb_1]*elem[rb_1];
                local_z += mz[rb_1]*elem[rb_1];

                local_x += mx[rb_2]*elem[rb_2];
                local_y += my[rb_2]*elem[rb_2];
                local_z += mz[rb_2]*elem[rb_2];

                local_x += mx[rb_3]*elem[rb_3];
                local_y += my[rb_3]*elem[rb_3];
                local_z += mz[rb_3]*elem[rb_3];
 
 
            }

            var_x[y+2*L*z] = local_x;
            var_y[y+2*L*z] = local_y;
            var_z[y+2*L*z] = local_z;
        }
    }


}