#pragma once

void init_fill(float * mx, float * my, float * mz, const grid_color color,
               const int L, trng::lcg64_shift * Rng, trng::uniform_dist<float> Uni);

void init_oc(int * oc, float * macr, float * mafe, const int L,
             trng::lcg64_shift * Rng, trng::uniform_dist<float> Uni,
             grid_color color);

void interac_init(int * oc, float * KA,
               float * Jx, float * Jy, float * Jz,
               float * Dx, float * Dy, float * Dz,
               const grid_color color, const int L);

void mc_init(int * oc, float * mc, const int L, grid_color color);