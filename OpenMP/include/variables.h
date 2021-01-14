#pragma once

#include <string>
#include <trng/lcg64_shift.hpp>
#include <trng/uniform_dist.hpp>
#include "omp.h"
#include <iostream>

#define IX(i,j,k) (idx((i), (j), (k), (2*L)))
#define RB(i,j,k) (rb_idx(i, j, k, L))
#define STRF(color,y,z) ((color + y&1 + z&1)&1);

typedef enum { RED_TILES, BLACK_TILES } grid_color;

static int MAX_T = 12;

static float frac;
static float flucfrac;

static int max_threads =  omp_get_max_threads();

static int t_mic;            // measuring interval
static int tmax;             // updates per Temperature
static const int Nsamp = 1;  // Samples fot statistics

static const float mfe = 1.0f;
static const float mcr = 0.6f;

static const float JJ = 1.0f;
static const float Jfefe = JJ;
static const float Jfecr = 153.0f/628.0f;
static const float Jcrcr = 115.0f/628.0f;
static const float Dfefe = 0.0214f*JJ;
static const float Dcrcr = 0.0074f*JJ;
static const float Dfecr = -0.0168f*JJ;

static const float Kcr = 0.007f*JJ;
static const float Kfe = 0.007f*JJ;

static const float Hfc = 0.002f*JJ;         // Z direction field
static const float Tempmax = 1.6f*JJ;
static const float Tempmin = 0.01f*JJ;
static const int Nptos = 100;            // Amount of Temperatures to simulate
static const float acep_rate = 0.46f;

static float Temp = 0.0f, H = 0.0f, Hx = 0.0f, Hy = 0.0f, Hz = 0.0f;

static const std::string ID = "123";    // Change this values for the reference that you prefer
static const std::string  archivofe  = "t_fe_mx_my_mz_m" + ID + ".dat";     // Temperature vs magnetization
static const std::string  archivocr  = "t_cr_mx_my_mz_m" + ID + ".dat";     // Temperature vs magnetization
static const std::string  archivo1   = "temp_mx_my_mz_m" + ID + ".dat";     // Temperature vs magnetization
static const std::string  archivo2   = "temp_mxs_mys_mzs_ms" + ID + ".dat"; // Temperature vs staggered magnetization
static const std::string  archivo3   = "temp_R" + ID + ".dat";              // Temperature vs acceptance ratio
static const std::string  fenergia   = "energia" + ID + ".dat";             // Energy Terms
static const std::string  fsuscep    = "susceptibilidad" + ID + ".dat";     // Susceptibility
static const std::string  fbinder    = "binder" + ID + ".dat";              // Binder Cumulant
static const std::string  fmag       = "magnetizacion" + ID + ".dat";       // Magnetization
static const std::string  pict       = "picture" + ID + ".dat";             // magnetica config

static const float pi = 3.14159265358979323846264338328;
static const float dospi = 2.*pi;

static inline int c(const int i, const int n) {
    return ((i % n) + n) % n;
}

static inline size_t idx(size_t i, size_t j, size_t k, size_t dim) {
    return (i + dim*(j + dim*k));
}

static inline float p(const float x, const int y) {
    return std::pow(x, y);
}

static inline size_t rb_idx(size_t i, size_t j, size_t k, size_t dim) {
    const int res = i + dim*(j + dim*2*k);
    return res;
}

static inline float myrand(trng::lcg64_shift * Rng, trng::uniform_dist<float> Uni
                           ,const int t_id){

    const float res = Uni(Rng[t_id]);
    return res;

}
