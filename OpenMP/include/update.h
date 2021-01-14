#pragma once

void update (const float * const mx, const float * const my, const float * const mz,
            const float * const KA, const float * mc, const float * const Jx_s,
            const float * const Jy_s, const float * const Jz_s,
            const float * const Dx_s, const float * const Dy_s,
            const float * const Dz_s, const float * const Jx_o,
            const float * const Jy_o, const float * const Jz_o,
            const float * const Dx_o, const float * const Dy_o,
            const float * const Dz_o, const grid_color color, bool * dirty,
            float * new_x, float * new_y, float * new_z,
            const float * const other_x, const float * const other_y,
            const float * const other_z, const int L,
            const float Hx, const float Hy, const float Hz, const float Temp,
            int * cont,trng::lcg64_shift * Rng, trng::uniform_dist<float> Uni);//, trng::lcg64_shift Rng, trng::uniform_dist<float> Uni);

