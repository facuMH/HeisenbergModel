#include <chrono>
#include <math.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <array>
#include <string>

#include "cub/cub.cuh"
#include <cub/util_allocator.cuh>

#include "vector_types.h"
#include "helper_cuda.h"
#include "omp.h"

#include "graphics.h"
#include "variables.h"
#include "update.h"
#include "energy.h"
#include "funs.h"
#include "main_kernels.h"


cub::CachingDeviceAllocator g_allocator(true);

#include <vector>

const std::string file_name = "spin_conf";

bool * dirty;
int * oc, * in1, * in2, * out;
float * macr_r, * macr_b, * mafe_r, * mafe_b;               // ocupation matrixes
float * mx_r, * mx_b, * my_r, *my_b, * mz_r, * mz_b;        // spin coordinates
float * mc_r, * mc_b;                                       // magenetization matrixes
float * Jz_r, * Jz_b, * Jy_r, * Jy_b, * Jx_r, * Jx_b;       // random interaction constants
float * Dz_r, * Dz_b, * Dy_r, * Dy_b, * Dx_r, * Dx_b;       // random interaction constants
float * KA_r, * KA_b;                                       // random interaction constants

// Auxiliar Matrixes
float * am1_x, * am1_y, * am1_z;
float * am2_x, * am2_y, * am2_z;
float * black_x, * black_y, * black_z;
float * red_x, * red_y, * red_z;
float * out_half;

void * d_ts_f = NULL;          // Temp Storage Float
void * d_ts_i = NULL;          // Temp Storage Int
void * d_half = NULL;          // Temp Storage RB
size_t d_ts_f_bytes = 0;
size_t d_ts_i_bytes = 0;
size_t d_half_bytes = 0;


static int L; // Size
static int N; // = 8*L*L*L;
static int RBN; // = 4*L*L*L;

static int t_mic;          // measuring interval
static int tmax;           // updates per Temperature

static int BX;
static int BY;
static int BZ;

static int GXf;
static int GX;
static int GY;
static int GZ;

static dim3 threads;
static dim3 blocks;
static dim3 blocksf; //block dim for full size matrix kernel

static int allocate_data (void)
{
    checkCudaErrors(cudaMallocManaged(&oc, N * sizeof(int)));

    checkCudaErrors(cudaMallocManaged(&in1, RBN * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&in2, RBN * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&out, RBN * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&out_half, RBN * sizeof(float)));

    checkCudaErrors(cudaMallocManaged(&macr_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&macr_b, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&mafe_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&mafe_b, RBN * sizeof(float)));

    checkCudaErrors(cudaMallocManaged(&Jx_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Jx_b, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Jy_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Jy_b, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Jz_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Jz_b, RBN * sizeof(float)));

    checkCudaErrors(cudaMallocManaged(&Dx_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Dx_b, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Dy_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Dy_b, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Dz_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&Dz_b, RBN * sizeof(float)));

    checkCudaErrors(cudaMallocManaged(&mx_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&mx_b, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&my_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&my_b, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&mz_r, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&mz_b, RBN * sizeof(float)));

    cudaMallocManaged(&mc_r, RBN * sizeof(float));
    cudaMallocManaged(&mc_b, RBN * sizeof(float));
    cudaMallocManaged(&KA_r, RBN * sizeof(float));
    cudaMallocManaged(&KA_b, RBN * sizeof(float));

    checkCudaErrors(cudaMallocManaged(&am1_x, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&am1_y, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&am1_z, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&am2_x, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&am2_y, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&am2_z, RBN * sizeof(float)));

    checkCudaErrors(cudaMallocManaged(&red_x, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&red_y, RBN* sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&red_z, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&black_x, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&black_y, RBN * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&black_z, RBN * sizeof(float)));

    CubDebugExit(cub::DeviceReduce::Sum(d_ts_i, d_ts_i_bytes, in1, out, N));
    CubDebugExit(g_allocator.DeviceAllocate(&d_ts_i, d_ts_i_bytes));

    CubDebugExit(cub::DeviceReduce::Sum(d_half, d_half_bytes, red_x, out_half, RBN));
    CubDebugExit(g_allocator.DeviceAllocate(&d_half, d_half_bytes));

  std::cout << "size: " << N << "\n";
  if (!macr_r || !macr_b || !mafe_r || !mafe_b || !mc_r || !mc_b || !oc ||
      !KA_r || !KA_b || !in1 || !in2 || !out  ||
      !mx_r || !mx_b || !my_r || !my_b || !mz_r || !mz_b ||
      !Jx_r || !Jx_b || !Jy_r || !Jy_b || !Jz_r || !Jz_b ||
      !Dx_r || !Dx_b || !Dy_r || !Dy_b || !Dz_r || !Dz_b ||
      !am1_x || !am1_y || !am1_z || !am2_x || !am2_y || !am2_z ||
      !d_ts_f || !d_ts_i || !d_half || !out_half ||
      !red_x || !black_x || !red_y || !black_y || !red_z || !black_z
      ){
    fprintf(stderr, "cannot allocate data\n");
    return 0;
  }
  return 1;
}

static void free_data(){
    std::cout << "in.. " << std::flush;
    if (in1) cudaFree(in1);
    if (in2) cudaFree(in2);
    std::cout << "out.. " << std::flush;
    if (out) cudaFree(out);
    std::cout << "out_half.. " << std::flush;
    if (out_half) cudaFree(out_half);

    std::cout << "oc.. " << std::flush;
    if (oc) cudaFree(oc);
    std::cout << "macr.. " << std::flush;
    if (macr_r) cudaFree(macr_r);
    if (macr_b) cudaFree(macr_b);
    std::cout << "mafe.. "<< std::flush;
    if (mafe_r) cudaFree(mafe_r);
    if (mafe_b) cudaFree(mafe_b);
    std::cout << "mc.. "<< std::flush;
    if (mc_r) cudaFree(mc_r);
    if (mc_b) cudaFree(mc_b);

    std::cout << "mx.. "<< std::flush;
    if (mx_r) cudaFree(mx_r);
    if (mx_b) cudaFree(mx_b);
    std::cout << "my.. "<< std::flush;
    if (my_r) cudaFree(my_r);
    if (my_b) cudaFree(my_b);
    std::cout << "mz.. "<< std::flush;
    if (mz_r) cudaFree(mz_r);
    if (mz_b) cudaFree(mz_b);

    std::cout << "Jx.. "<< std::flush;
    if (Jx_r) cudaFree(Jx_r);
    if (Jx_b) cudaFree(Jx_b);
    std::cout << "Jy.. "<< std::flush;
    if (Jy_r) cudaFree(Jy_r);
    if (Jy_b) cudaFree(Jy_b);
    std::cout << "Jz.. "<< std::flush;
    if (Jz_r) cudaFree(Jz_r);
    if (Jz_b) cudaFree(Jz_b);

    std::cout << "Dx.. "<< std::flush;
    if (Dx_r) cudaFree(Dx_r);
    if (Dx_b) cudaFree(Dx_b);
    std::cout << "Dy.. "<< std::flush;
    if (Dy_r) cudaFree(Dy_r);
    if (Dy_b) cudaFree(Dy_b);
    std::cout << "Dz.. "<< std::flush;
    if (Dz_r) cudaFree(Dz_r);
    if (Dz_b) cudaFree(Dz_b);

    std::cout << "KA.. " << std::flush;
    if (KA_r) cudaFree(KA_r);
    if (KA_b) cudaFree(KA_b);

    std::cout << "am1_x.. " << std::flush;
    if (am1_x) cudaFree(am1_x);
    std::cout << "am1_y .. " << std::flush;
    if (am1_y) cudaFree(am1_y);
    std::cout << "am1_z.. " << std::flush;
    if (am1_z) cudaFree(am1_z);
    std::cout << "am2_x.. " << std::flush;

    if (am2_x) cudaFree(am2_x);
    std::cout << "am2_y.. " << std::flush;
    if (am2_y) cudaFree(am2_y);
    std::cout << "am2_z.. " << std::flush;
    if (am2_z) cudaFree(am2_z);

    std::cout << "ts_f.. " << std::flush;
    if (d_ts_f) CubDebugExit(g_allocator.DeviceFree(d_ts_f));
    std::cout << "ts_i.. " << std::flush;
    if (d_ts_i) CubDebugExit(g_allocator.DeviceFree(d_ts_i));
    std::cout << "ts_half.. " << std::flush;
    if (d_half) CubDebugExit(g_allocator.DeviceFree(d_half));

    std::cout << "red_x.. " << std::flush;
    if (red_x) cudaFree(red_x);
    std::cout << "red_y.. " << std::flush;
    if (red_y) cudaFree(red_y);
    std::cout << "red_z.. " << std::flush;
    if (red_z) cudaFree(red_z);

    std::cout << "black_x.. " << std::flush;
    if (black_x) cudaFree(black_x);
    std::cout << "black_y.. " << std::flush;
    if (black_y) cudaFree(black_y);
    std::cout << "black_z.. " << std::flush;
    if (black_z) cudaFree(black_z);

    std::cout << "\n";
};

/*
This function defines the element for each site of the cubic matrix
*/
void set_interac(int * oc, float * Jx_r, float * Jx_b, float * Jy_r,
                 float * Jy_b, float * Jz_r, float * Jz_b,
                 float * Dx_r, float * Dx_b, float * Dy_r, float * Dy_b,
                 float * Dz_r, float * Dz_b, float * KA_r, float * KA_b,
                 float * macr_r, float* macr_b, float * mafe_r, float * mafe_b){

    unsigned int _time = static_cast<unsigned int> (time(NULL));
    init_oc<<<blocksf, threads>>>(oc, macr_r, mafe_r, L, _time, frac, flucfrac, RED_TILES);
    getLastCudaError("init_oc failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    init_oc<<<blocksf, threads>>>(oc, macr_b, mafe_b, L, _time, frac, flucfrac, BLACK_TILES);
    getLastCudaError("init_oc failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    interac_init<<<blocks, threads>>>(oc, KA_r, Jx_r, Jy_r, Jz_r,
                   Dx_r, Dy_r, Dz_r, L, RED_TILES);
    getLastCudaError("interac_init RED failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    interac_init<<<blocks, threads>>>(oc, KA_b, Jx_b, Jy_b, Jz_b,
                   Dx_b, Dy_b, Dz_b, L, BLACK_TILES);
    getLastCudaError("interac_init BLACK failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

}


int main(int argc, char * argv[]){
    bool save_files = false;
    bool graphics = false;
    bool size = false;
    bool dims = false;
    bool temps = false;

    while (*++argv){
      if ((*argv)[0] == '-'){
        switch ((*argv)[1]){
          case 'h':
              std::cout << "\n\t-g for graphic.\n\t-o for file output.";
              std::cout << "\n\t-L X where X is an integer for the size";
              std::cout << "considering N=(2*L)^3";
              std::cout << "\n\t-B BX BY BZ for the block dimensions";
              std::cout << "\n\t-t for the t_mic tmax parameters";
			  std::cout << "\n\t-f for the frac and flucfrac parameters";
              exit(0);
          case 'o':
              save_files = true;
              std::cout << "\n\tFile output enables";
              break;
          case 'g':
              graphics = true;
              std::cout << "\n\tGraphics representation activated";
              break;
          case 'L':
              L = atoi(argv[1]);
              N = 8*L*L*L;
              RBN = 4*L*L*L;
              size = true;
              std::cout << "\n\t Size detected. L:"<<L;
              break;
          case 'B':
              BX = atoi(argv[1]);
              BY = atoi(argv[2]);
              BZ = atoi(argv[3]);
              threads = dim3(BX,BY,BZ);
              dims = true;
              std::cout << "\n\t BY:" << BY <<" BZ:" << BZ;
              break;
          case 't':
              t_mic = atoi(argv[1]);
              tmax = atoi(argv[2]);
              temps = true;
              std::cout << "\n\tt_mic:"<<t_mic<<" and tmax:"<<tmax;
              break;
          case 'f':
              frac = std::stof(argv[1]);
              flucfrac = std::stof(argv[2]);
              std::cout << "\n\tfrac:"<< frac <<" and flucfrac:"<<flucfrac ;
              break;
          default:
              std::cout << "\n\tUnknown option -\n\n" << (*argv)[1] << "\n";
              break;
        }
      }
    }

    if (!size ){
      std::cout << "Please run -h and make sure you input -L -B and -t\n";
      exit(1);
    }

    if (!dims){
        BX = 16;
        BY = 2;
        BZ = 1;
        threads = dim3(BX,BY,BZ);
    }
    if (!temps){
        t_mic = 100;
        tmax = 1000;
    }

    //grid dimensions
    GXf = ceil(2*L/(float)BX);
    GX = ceil(L/(float)BX);
    GY = ceil(2*L/(float)BY);
    GZ = ceil(2*L/(float)BZ);
    blocks = dim3(GX,GY,GZ);
    blocksf = dim3(GXf,GY,GZ);

    int ll1,ll3;
    unsigned int t =0;
    float R;
    float s_mx, s_my, s_mz;
    float s_mx_1, s_my_1, s_mz_1;
    float s_mx_2, s_my_2, s_mz_2;

    float cr_mx, cr_my, cr_mz;
    float cr_mx_1, cr_my_1, cr_mz_1;
    float cr_mx_2, cr_my_2, cr_mz_2;
    float fe_mx, fe_my, fe_mz;
    float fe_mx_1, fe_my_1, fe_mz_1;
    float fe_mx_2, fe_my_2, fe_mz_2;
    float mag;

    float ss_mx[Nptos], ss_my[Nptos], ss_mz[Nptos];
    float ss_mx_1[Nptos], ss_my_1[Nptos], ss_mz_1[Nptos];
    float ss_mx_2[Nptos], ss_my_2[Nptos], ss_mz_2[Nptos];

    float ss_m[Nptos], ss_m2[Nptos], ss_m4[Nptos];
    float ss_m_1[Nptos], ss_m_2[Nptos], temp_a[Nptos], R_a[Nptos];
    float ss_U[Nptos], ss_U2[Nptos];

    float sfe_mx[Nptos], sfe_my[Nptos], sfe_mz[Nptos];
    float sfe_mx_1[Nptos], sfe_my_1[Nptos], sfe_mz_1[Nptos];
    float sfe_mx_2[Nptos], sfe_my_2[Nptos], sfe_mz_2[Nptos];
    float scr_mx[Nptos], scr_my[Nptos], scr_mz[Nptos];
    float scr_mx_1[Nptos], scr_my_1[Nptos], scr_mz_1[Nptos];
    float scr_mx_2[Nptos], scr_my_2[Nptos], scr_mz_2[Nptos];

    float tmp1_x, tmp1_y, tmp1_z;
    float tmp2_x, tmp2_y, tmp2_z;

    float cr1_x, cr1_y, cr1_z;
    float cr2_x, cr2_y, cr2_z;
    float fe1_x, fe1_y, fe1_z;
    float fe2_x, fe2_y, fe2_z;

    float E_afm_ex, E_afm_an, E_afm_dm, U, U2, m1, m2, m4;
    float Temp,H, Hx, Hy, Hz;

    // time sampling vars
    double start = 0.0;
    double stop = 0.0;
    double nseconds = 0.0;
    double total = 0.0;
    double high = 0.0;
    double low = 0.0;
    double aux_termal = 0.0;
    double aux_second = 0.0;
    double total_time = 0.0;

    // accepted changes accumulator
    int suma = 0;

    allocate_data();

    std::cout << "\nArgs: " << argc<< "\n";

    ll1 = 0;
    ll3 = 0;

    std::cout << "Today N is: " << N << "\n";
    set_interac(oc, Jx_r, Jx_b, Jy_r, Jy_b, Jz_r, Jz_b, Dx_r, Dx_b, Dy_r, Dy_b,
                Dz_r, Dz_b, KA_r, KA_b, macr_r, macr_b, mafe_r, mafe_b);

    mc_init<<<blocksf, threads>>>(oc, mc_r, L, RED_TILES);
    getLastCudaError("mc_init red failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    mc_init<<<blocksf, threads>>>(oc, mc_b, L, BLACK_TILES);
    getLastCudaError("mc_init black failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "\n";
    std::cout << "initialized mc" << "\n";
    std::cout << "\n";

    MyApp app;
    if (graphics){
      app.App_init();
      app.init(L);
    }

    H = Hfc;
    Hx = 0.0f;
    Hy = 0.0f;
    Hz = H;

    std::fill(std::begin(ss_mx), std::end(ss_mx), 0.0f);
    std::fill(std::begin(ss_my), std::end(ss_my), 0.0f);
    std::fill(std::begin(ss_mz), std::end(ss_mz), 0.0f);

    std::fill(std::begin(ss_mx_1), std::end(ss_mx_1), 0.0f);
    std::fill(std::begin(ss_my_1), std::end(ss_my_1), 0.0f);
    std::fill(std::begin(ss_mz_1), std::end(ss_mz_1), 0.0f);

    std::fill(std::begin(ss_mx_2), std::end(ss_mx_2), 0.0f);
    std::fill(std::begin(ss_my_2), std::end(ss_my_2), 0.0f);
    std::fill(std::begin(ss_mz_2), std::end(ss_mz_2), 0.0f);

    std::fill(std::begin(scr_mx), std::end(scr_mx), 0.0f);
    std::fill(std::begin(scr_my), std::end(scr_my), 0.0f);
    std::fill(std::begin(scr_mz), std::end(scr_mz), 0.0f);

    std::fill(std::begin(scr_mx_1), std::end(scr_mx_1), 0.0f);
    std::fill(std::begin(scr_my_1), std::end(scr_my_1), 0.0f);
    std::fill(std::begin(scr_mz_1), std::end(scr_mz_1), 0.0f);

    std::fill(std::begin(scr_mx_2), std::end(scr_mx_2), 0.0f);
    std::fill(std::begin(scr_my_2), std::end(scr_my_2), 0.0f);
    std::fill(std::begin(scr_mz_2), std::end(scr_mz_2), 0.0f);

    std::fill(std::begin(ss_m), std::end(ss_m), 0.0f);
    std::fill(std::begin(ss_m2), std::end(ss_m2), 0.0f);
    std::fill(std::begin(ss_m4), std::end(ss_m4), 0.0f);
    std::fill(std::begin(ss_m_1), std::end(ss_m_1), 0.0f);
    std::fill(std::begin(ss_m_2), std::end(ss_m_2), 0.0f);
    std::fill(std::begin(R_a), std::end(R_a), 0.0f);

    std::fill(std::begin(ss_U), std::end(ss_U), 0.0f);
    std::fill(std::begin(ss_U2), std::end(ss_U2), 0.0f);

    unsigned int aux_temp = 0;
    int counter = 0;

    total_time = omp_get_wtime();

    for (ll3 = 0; ll3 < Nsamp; ll3++){
          Temp = Tempmax;
          //initial random spin configuration.
          t += static_cast<unsigned int> (time(NULL));
          init_fill<<<blocks, threads>>>(mx_r, my_r, mz_r, L, t, RED_TILES);
          getLastCudaError("init fill RED failed\n");
          checkCudaErrors(cudaDeviceSynchronize());

          t += static_cast<unsigned int> (time(NULL));
          init_fill<<<blocks, threads>>>(red_x, red_y, red_z, L, t, RED_TILES);
          getLastCudaError("init fill RED failed\n");
          checkCudaErrors(cudaDeviceSynchronize());


          t += static_cast<unsigned int> (time(NULL));
          init_fill<<<blocks, threads>>>(mx_b, my_b, mz_b, L, t, BLACK_TILES);
          getLastCudaError("init fill BLACK failed\n");
          checkCudaErrors(cudaDeviceSynchronize());

          t += static_cast<unsigned int> (time(NULL));
          init_fill<<<blocks, threads>>>(black_x, black_y, black_z, L, t, BLACK_TILES);
          getLastCudaError("init fill BLACK failed\n");
          checkCudaErrors(cudaDeviceSynchronize());


          std::cout << "initialized mx my mz" << "\n";

          H = Hfc;

          for (ll1 = 0; ll1 < Nptos; ll1++){

                std::cout << " round: " << ll1 << "  - ";
                Temp -= (Tempmax - Tempmin)/Nptos;

                aux_second = 0.0;
                aux_termal = 0.0;

                R = 0.0f;
                mag = 0.0f;
                m1 = 0.0f;
                m2 = 0.0f;
                m4 = 0.0f;

                U = 0.0f;
                U2 = 0.0f;

                s_mx = 0.0f;
                s_my = 0.0f;
                s_mz = 0.0f;

                s_mx_1 = 0.0f;
                s_my_1 = 0.0f;
                s_mz_1 = 0.0f;

                s_mx_2 = 0.0f;
                s_my_2 = 0.0f;
                s_mz_2 = 0.0f;

                cr_mx = 0.0f;
                cr_my = 0.0f;
                cr_mz = 0.0f;

                cr_mx_1 = 0.0f;
                cr_my_1 = 0.0f;
                cr_mz_1 = 0.0f;

                cr_mx_2 = 0.0f;
                cr_my_2 = 0.0f;
                cr_mz_2 = 0.0f;

                fe_mx = 0.0f;
                fe_my = 0.0f;
                fe_mz = 0.0f;

                fe_mx_1 = 0.0f;
                fe_my_1 = 0.0f;
                fe_mz_1 = 0.0f;

                fe_mx_2 = 0.0f;
                fe_my_2 = 0.0f;
                fe_mz_2 = 0.0f;

                suma = 0;
                counter = 0;
                // stabilizing system to the 'new' temperature
                double stage[tmax];
                for (t = 0; t < tmax; t++){
                    aux_temp += static_cast<unsigned int> (time(NULL));;

                    start = omp_get_wtime();
                    update(mx_r, my_r, mz_r, mx_b, my_b, mz_b,
                           Jx_r, Jy_r, Jz_r, Dx_r, Dy_r, Dz_r,
                           Jx_b, Jy_b, Jz_b, Dx_b, Dy_b, Dz_b,
                           L, KA_r, KA_b, mc_r, mc_b, Hx, Hy, Hz, Temp,
                           aux_temp, in1, in2,
                           blocks, threads,
                           red_x, red_y, red_z, black_x, black_y, black_z);
                    stop = omp_get_wtime();

                    swap(mx_r,red_x);
                    swap(my_r,red_y);
                    swap(mz_r,red_z);

                    swap(mx_b,black_x);
                    swap(my_b,black_y);
                    swap(mz_b,black_z);

                    CubDebugExit(cub::DeviceReduce::Sum(d_ts_i, d_ts_i_bytes, in1, out, RBN));
                    getLastCudaError("update reduct 1 failed\n");
                    checkCudaErrors(cudaDeviceSynchronize());

                    assert(out[0] <= N);
                    suma += out[0];

                    CubDebugExit(cub::DeviceReduce::Sum(d_ts_i, d_ts_i_bytes, in2, out, RBN));
                    getLastCudaError("update reduct 2 failed\n");
                    checkCudaErrors(cudaDeviceSynchronize());

                    assert(out[0] <= N);
                    suma += out[0];

                    counter += 1;
                    stage[t] = 1.0e9*(stop - start)/N;
                    aux_termal += stage[t];

                    if (graphics) {
                        app.drawOne(mx_r, my_r, mz_r, mx_b, my_b, mz_b);
                    }
                }

                std::cout << " average: " << (suma/(float)counter)/(float)N;
                suma = 0;
                counter = 0;

                for (t = 0; t < tmax; t++){
                    aux_temp += static_cast<unsigned int> (time(NULL));

                    start = omp_get_wtime();
                    update(mx_r, my_r, mz_r, mx_b, my_b, mz_b,
                           Jx_r, Jy_r, Jz_r, Dx_r, Dy_r, Dz_r,
                           Jx_b, Jy_b, Jz_b, Dx_b, Dy_b, Dz_b,
                           L, KA_r, KA_b, mc_r, mc_b, Hx, Hy, Hz,
                           Temp, aux_temp, in1, in2,
                           blocks, threads,
                           red_x, red_y, red_z, black_x, black_y, black_z);
                    stop = omp_get_wtime();

                    swap(mx_r,red_x);
                    swap(my_r,red_y);
                    swap(mz_r,red_z);

                    swap(mx_b,black_x);
                    swap(my_b,black_y);
                    swap(mz_b,black_z);

                    CubDebugExit(cub::DeviceReduce::Sum(d_ts_i, d_ts_i_bytes, in1, out, RBN));
                    getLastCudaError("update reduct 3 failed\n");
                    checkCudaErrors(cudaDeviceSynchronize());

                    assert(out[0] <= N);
                    suma += out[0];

                    CubDebugExit(cub::DeviceReduce::Sum(d_ts_i, d_ts_i_bytes, in2, out, RBN));
                    getLastCudaError("update reduct 4 failed\n");
                    checkCudaErrors(cudaDeviceSynchronize());

                    assert(out[0] <= N);
                    suma += out[0];

                    counter += 1;

                    R += suma;
                    stage[t] = 1.0e9*(stop - start)/N;
                    aux_second += stage[t];

                    if (graphics){
                        app.drawOne(mx_r, my_r, mz_r, mx_b, my_b, mz_b);
                    }

                    if (t%t_mic == 0){
                        tmp1_x = 0.0f;
                        tmp1_y = 0.0f;
                        tmp1_z = 0.0f;

                        tmp2_x = 0.0f;
                        tmp2_y = 0.0f;
                        tmp2_z = 0.0f;

                        cr1_x = 0.0f;
                        cr1_y = 0.0f;
                        cr1_z = 0.0f;

                        cr2_x = 0.0f;
                        cr2_y = 0.0f;
                        cr2_z = 0.0f;

                        fe1_x = 0.0f;
                        fe1_y = 0.0f;
                        fe1_z = 0.0f;

                        fe2_x = 0.0f;
                        fe2_y = 0.0f;
                        fe2_z = 0.0f;


                        calculate(&tmp1_x, &tmp1_y, &tmp1_z,
                                  &tmp2_x, &tmp2_y, &tmp2_z,
                                  mx_r, my_r, mz_r,
                                  mx_b, my_b, mz_b, mc_r, mc_b,
                                  // the following are used as auxiliary matrixes
                                  am1_x, am1_y, am1_z,
                                  am2_x, am2_y, am2_z,
                                  out_half, d_half, d_half_bytes,
                                  L, blocks, threads);

                        calculate(&cr1_x, &cr1_y, &cr1_z,
                                  &cr2_x, &cr2_y, &cr2_z,
                                  mx_r, my_r, mz_r,
                                  mx_b, my_b, mz_b, macr_r, macr_b,
                                  am1_x, am1_y, am1_z,
                                  am2_x, am2_y, am2_z,
                                  out_half, d_half, d_half_bytes,
                                  L, blocks, threads);

                        calculate(&fe1_x, &fe1_y, &fe1_z,
                                  &fe2_x, &fe2_y, &fe2_z,
                                  mx_r, my_r, mz_r,
                                  mx_b, my_b, mz_b, mafe_r, mafe_b,
                                  am1_x, am1_y, am1_z,
                                  am2_x, am2_y, am2_z,
                                  out_half, d_half, d_half_bytes,
                                  L, blocks, threads);

                        fe_mx += mfe*(fe1_x + fe2_x)/N;
                        fe_my += mfe*(fe1_y + fe2_y)/N;
                        fe_mz += mfe*(fe1_z + fe2_z)/N;

                        fe_mx_1 += mfe*abs(fe1_x)/(N);
                        fe_my_1 += mfe*abs(fe1_y)/(N);
                        fe_mz_1 += mfe*abs(fe1_z)/(N);

                        fe_mx_2 += mfe*abs(fe2_x)/(N);
                        fe_my_2 += mfe*abs(fe2_y)/(N);
                        fe_mz_2 += mfe*abs(fe2_z)/(N);

                        cr_mx += mcr*(cr1_x + cr2_x)/N;
                        cr_my += mcr*(cr1_y + cr2_y)/N;
                        cr_mz += mcr*(cr1_z + cr2_z)/N;

                        cr_mx_1 += mcr*abs(cr1_x)/(N);
                        cr_my_1 += mcr*abs(cr1_y)/(N);
                        cr_mz_1 += mcr*abs(cr1_z)/(N);

                        cr_mx_2 += mcr*abs(cr2_x)/(N);
                        cr_my_2 += mcr*abs(cr2_y)/(N);
                        cr_mz_2 += mcr*abs(cr2_z)/(N);

                        s_mx += (tmp1_x + tmp2_x)/N;
                        s_my += (tmp1_y + tmp2_y)/N;
                        s_mz += (tmp1_z + tmp2_z)/N;

                        s_mx_1 += abs(tmp1_x)/N;
                        s_my_1 += abs(tmp1_y)/N;
                        s_mz_1 += abs(tmp1_z)/N;

                        s_mx_2 += abs(tmp2_x)/N;
                        s_my_2 += abs(tmp2_y)/N;
                        s_mz_2 += abs(tmp2_z)/N;

                        mag += (((tmp1_x + tmp2_x)/N)*Hx +
                                ((tmp1_y + tmp2_y)/N)*Hy +
                                ((tmp1_z + tmp2_z)/N)*Hz)/H;

                        m1 += 2.0f*sqrt((tmp1_x/N)*(tmp1_x/N) +
                                        (tmp1_y/N)*(tmp1_y/N) +
                                        (tmp1_z/N)*(tmp1_z/N));

                        m2  += 4.0f*(   (tmp1_x/N*tmp1_x/N) +
                                        (tmp1_y/N*tmp1_y/N) +
                                        (tmp1_z/N*tmp1_z/N));

                        m4  += 16.0f*(  (tmp1_x/N*tmp1_x/N) +
                                        (tmp1_y/N*tmp1_y/N) +
                                        (tmp1_z/N*tmp1_z/N))*
                                     (  (tmp1_x/N*tmp1_x/N) +
                                        (tmp1_y/N*tmp1_y/N) +
                                        (tmp1_z/N*tmp1_z/N));

                        energy(mx_r, my_r, mz_r, mx_b, my_b, mz_b,
                               &E_afm_ex, &E_afm_an, &E_afm_dm, KA_r, KA_b,
                               mc_r, mc_b,
                               Jx_r, Jy_r, Jz_r, Jx_b, Jy_b, Jz_b,
                               Dx_r, Dy_r, Dz_r, Dx_b, Dy_b, Dz_b,
                               Hx, Hy, Hz,
                               am1_x, am1_y, am1_z,
                               am2_x, am2_y, am2_z,
                               out_half, d_half, d_half_bytes,
                               L, blocks, threads);

                        U += E_afm_ex + E_afm_an + E_afm_dm;
                        U2 += (E_afm_ex + E_afm_an + E_afm_dm)*(E_afm_ex + E_afm_an + E_afm_dm);
                    }
                }

            suma = 0;
            counter = 0;


            nseconds = (aux_termal + aux_second)/(tmax*2);
            high = high > nseconds ? high : nseconds;
            if (!low){
                low = nseconds;
            } else {
                low = nseconds > low ? low : nseconds;
            }

            std::cout << " partial - nsec: " << nseconds;
            total += nseconds;
            nseconds = 0;

            temp_a[ll1] = Temp;

            ss_mx[ll1] += (s_mx*t_mic/tmax);
            ss_my[ll1] += (s_my*t_mic/tmax);
            ss_mz[ll1] += (s_mz*t_mic/tmax);

            ss_mx_1[ll1] += (2.0f*s_mx_1*t_mic/tmax);
            ss_my_1[ll1] += (2.0f*s_my_1*t_mic/tmax);
            ss_mz_1[ll1] += (2.0f*s_mz_1*t_mic/tmax);

            ss_mx_2[ll1] += (2.0f*s_mx_2*t_mic/tmax);
            ss_my_2[ll1] += (2.0f*s_my_2*t_mic/tmax);
            ss_mz_2[ll1] += (2.0f*s_mz_2*t_mic/tmax);

            scr_mx[ll1] += (cr_mx*t_mic/tmax);
            scr_my[ll1] += (cr_my*t_mic/tmax);
            scr_mz[ll1] += (cr_mz*t_mic/tmax);

            scr_mx_1[ll1] += (2.0f*cr_mx_1*t_mic/tmax);
            scr_my_1[ll1] += (2.0f*cr_my_1*t_mic/tmax);
            scr_mz_1[ll1] += (2.0f*cr_mz_1*t_mic/tmax);

            scr_mx_2[ll1] += (2.0f*cr_mx_2*t_mic/tmax);
            scr_my_2[ll1] += (2.0f*cr_my_2*t_mic/tmax);
            scr_mz_2[ll1] += (2.0f*cr_mz_2*t_mic/tmax);

            sfe_mx[ll1] += (fe_mx*t_mic/tmax);
            sfe_my[ll1] += (fe_my*t_mic/tmax);
            sfe_mz[ll1] += (fe_mz*t_mic/tmax);

            sfe_mx_1[ll1] += (2.0f*fe_mx_1*t_mic/tmax);
            sfe_my_1[ll1] += (2.0f*fe_my_1*t_mic/tmax);
            sfe_mz_1[ll1] += (2.0f*fe_mz_1*t_mic/tmax);

            sfe_mx_2[ll1] += (2.0f*fe_mx_2*t_mic/tmax);
            sfe_my_2[ll1] += (2.0f*fe_my_2*t_mic/tmax);
            sfe_mz_2[ll1] += (2.0f*fe_mz_2*t_mic/tmax);

            ss_m_2[ll1] += 2.0f*sqrt((s_mx_2*s_mx_2) +
                                     (s_my_2*s_my_2) +
                                     (s_mz_2*s_mz_2) )
                               *t_mic/tmax;
            ss_m_1[ll1] += m1*t_mic/tmax;

            std::cout << " - ss_m_1["<< ll1<< "]:" << ss_m_1[ll1];
            std::cout << " - ss_m_2["<< ll1<< "]:" << ss_m_2[ll1] << "\n";

            ss_m2[ll1] += m2*t_mic/tmax;
            ss_m4[ll1] += m4*t_mic/tmax;

            ss_m[ll1] += mag*t_mic/tmax;
            ss_U[ll1] += U*t_mic/tmax;
            ss_U2[ll1] += U2*t_mic/tmax;

            R_a[ll1] += R/tmax*N;

        }

    }


    std::cout << "\n";

    total_time = omp_get_wtime() - total_time;
    std::cout << "TOTAL TIME: " << total_time/60 << "\n";
    std::cout << "AVERAGE = " << total/Nptos << "\n";
    std::cout << "HIGH = " << high << "\n";
    std::cout << "LOW = " << low << "\n" << "\n";

    if (save_files) {
      std::cout << " saving output" << "\n";
      // Output files
      // "t_fe_mx_my_mz_m"//ID//".dat"
      std::ofstream file_fe;
      file_fe.open (archivofe);
      file_fe << "# Nsamp" << " temp_a " << " sfe_mx " << " sfe_my " <<" sfe_mz " << "\n";
      // "t_cr_mx_my_mz_m"//ID//".dat"
      std::ofstream file_cr;
      file_cr.open (archivocr);
      file_cr << "# Nsamp" << " temp_a " << " scr_mx " << " scr_my " <<" scr_mz " << "\n";
      // "temp_mx_my_mz_m"//ID//".dat"
      std::ofstream file_1;
      file_1.open (archivo1);
      file_1 << "# Nsamp" << " ss_mx_1 " << " ss_my_1 " << " ss_mz_1"  << "ss_m_1" << "\n";
      // "temp_mxs_mys_mzs_ms"//ID//".dat"
      std::ofstream file_2;
      file_2.open (archivo2);
      file_2 << "# Nsamp" << " ss_mx_2 " << " ss_my_2 " << " ss_mz_2" << "\n";
      // "energia"//ID//".dat"
      std::ofstream file_energia;
      file_energia.open (fenergia);
      file_energia << "# Nsamp" << " ss_U" << "   " << "ss_U2" << "   " << "N*(ss_U2-ss_U^2)/(temp_a[i]^2)" << "\n";
      // "susceptibilidad"//ID//".dat"
      std::ofstream file_suscep;
      file_suscep.open (fsuscep);
      file_suscep << "# Nsamp" << "   " << "ss_m" << "  " << "ss_m2" << " " << "N*(ss_m2-ss_m_1^2)/(temp_a[i])"  << "\n";
      // "binder"//ID//".dat"
      std::ofstream file_binder;
      file_binder.open (fbinder);
      file_binder << "# Nsamp" << "   " << "ss_m2" << "   " << "ss_m4" << "   " << "1-(ss_m4/(3*ss_m2^2))" << "\n";
      // "magnetizacion"//ID//".dat"
      std::ofstream file_mag;
      file_mag.open (fmag);
      file_mag << "campo" << "   " << "ss_mx" << "   " << "ss_my" << "   " << "ss_mz" << "   " << "ss_m"<< "\n";
      std::ofstream last_state;
      last_state.open(std::string("last_state"));
      last_state << "   " << "i " << "   " << "j" << "   " << "k" << "   " << "mx" << "   " << "my" << "   " << "mz" << "\n";


      for (int i=0; i<Nptos; ++i){

        file_fe << "   " << temp_a[i] << "   " << sfe_mx[i]/ll3 << "   " << sfe_my[i]/ll3 << "   " << sfe_mz[i]/ll3 << "\n";
        file_cr << "   " << temp_a[i] << "   " << scr_mx[i]/ll3 << "   " << scr_my[i]/ll3 << "   " << scr_mz[i]/ll3 << "\n";
        file_1 << "   " << temp_a[i] << "   " << ss_mx_1[i]/ll3 << "   " << ss_my_1[i]/ll3 << "   " << ss_mz_1[i]/ll3 << "   " << ss_m_1[i]/ll3 << "\n";
        file_2 << "   " << temp_a[i] << "   " << ss_mx_2[i]/ll3 << "   " << ss_my_2[i]/ll3 << "   " << ss_mz_2[i]/ll3 << "   " << ss_m_2[i]/ll3 << "\n";
        file_energia << "   " << temp_a[i] << "   " << ss_U[i]/ll3 << "   " << ss_U2[i]/ll3 << "   " << N*(ss_U2[i] - p(ss_U[i],2))/(ll3*p(temp_a[i],2)) << "\n";
        file_suscep << "   "  << temp_a[i] << "   " << ss_m[i]/ll3 << "   " << ss_m2[i]/ll3 << "   " << N*(ss_m2[i] - p(ss_m_1[i],2))/(ll3*temp_a[i]) << "\n";
        file_binder << "   " << temp_a[i] << "   " << ss_m2[i]/ll3 << "   " << ss_m4[i]/ll3 << "   " << 1 - (ss_m4[i]/(3*p(ss_m2[i],2)))/(ll3) << "\n";
        file_mag << "   " << temp_a[i] << "   " << ss_mx[i]/ll3 << "   " << ss_my[i]/ll3 << "   " << ss_mz[i]/ll3 << "   " << ss_m[i]/ll3 << "\n";

      }

      int ixf = 0;
      int rbx = 0;
      for (int k=0; k<2*L; k++){
          for (int j=0; j<2*L; j++){
              for (int i=0; i<L; ++i){
                  ixf = 2*i + STRF(RED_TILES,j,k);   // i on full size matrix
                  rbx = RB(i,j,k);
                  last_state << "   " << ixf << "    " << j << "   " << k << "   " << mx_r[rbx] << "   " << my_r[rbx] << "   " << mz_r[rbx] << "\n";

                  ixf = 2*i + STRF(BLACK_TILES,j,k);   // i on full size matrix
                  rbx = RB(i,j,k);
                  last_state << "   " << ixf << "    " << j << "   " << k << "   " << mx_b[rbx] << "   " << my_b[rbx] << "   " << mz_b[rbx] << "\n";
              }
          }
      }


      file_fe.close();
      file_cr.close();
      file_1.close();
      file_2.close();
      file_energia.close();
      file_suscep.close();
      file_binder.close();
      file_mag.close();
      last_state.close();
    }


    if (graphics){
        string anything;
        std::cout << "press anything to end:";
        while(true){app.drawOne(mx_r, my_r, mz_r, mx_b, my_b, mz_b);}
    }

    free_data();

    return 0;

}
