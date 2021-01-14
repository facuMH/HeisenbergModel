

__global__
void init_fill(float * mx, float * my, float * mz, const int L,
               const unsigned int time, const grid_color color);

__global__
void init_oc(int * oc, float * macr, float * mafe, const int L,
             const unsigned int t, const float frac, const float fluecfrac,
             const grid_color color);

__global__
void interac_init(int * oc, float * KA, float * Jx, float * Jy, float * Jz,
               float * Dx, float * Dy, float * Dz, const int L,
               const grid_color color);


__global__
void mc_init(int * oc, float * mc, const int L, const grid_color color);
