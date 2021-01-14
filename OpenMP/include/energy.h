#pragma once
#ifndef ENERGY_H_INCLUDED
#define ENERGY_H_INCLUDED

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
            const float * const other_z, const grid_color color,
            const int L);

#endif