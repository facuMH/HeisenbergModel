#pragma once

#include <string>
#include <cmath>
#include "helper_cuda.h"
#include "vector_types.h"

typedef enum { RED_TILES, BLACK_TILES } grid_color;

#define IX(i,j,k) idx(i, j, k, 2*L)
#define RB(i,j,k) rb_idx(i, j, k, L)
#define STRF(color,y,z) ((color + y%2 + z%2)%2)

#define PI 3.14159265358979323846264338328f

static float frac = 0.55f;
static float flucfrac = 0.35f;

static const int Nsamp = 1;             // Samples for statistics

const float mfe = 1.0f;              //-0.0168*JJ
const float mcr = 0.6f;              //-0.0168*JJ

static const float JJ = 1.0f;
static const float Hfc = 0.002f*JJ;         // Z direction field
static const float Tempmax = 1.6f*JJ;
static const float Tempmin = 0.01f*JJ;
static const int Nptos = 100;            // Amount of Temperatures to simulate
static const float acep_rate = 0.46f;

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


__device__ __host__ static inline int c(const int i, const int n) {
    return ((i % n) + n) % n;
}

__device__ __host__ static inline size_t idx(const size_t i, const size_t j, const size_t k, const size_t dim) {
    return (i + dim*(j + dim*k));
}

__device__ __host__ static inline float p(const float x, const int y) {
    return powf(x, y);
}

__device__ __host__ static inline size_t rb_idx(const size_t i, const size_t j, const size_t k, const size_t dim) {
    return  i + dim*(j + dim*2*k);
}
