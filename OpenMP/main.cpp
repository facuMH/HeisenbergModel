#include <random>
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
#include "omp.h"

#include "variables.h"
#include "funs.h"
#include "main_kernels.h"
#include "update.h"
#include "energy.h"
#include "graphics.h"

const std::string file_name = "spin_conf";

int L; // Size
int N; // = 8*L*L*L;
int RBN; // = 4*L*L*L;

int * oc;
float * macr_r, * macr_b, * mafe_r, * mafe_b;     // ocupation matrixes
float * KA_r, * KA_b;                             // random interaction constants
float * mc_r, * mc_b;                             // magenetization matrixes
float * mx_r, * my_r, * mz_r;                     // spin coordinates
float * mx_b, * my_b, * mz_b;                     // spin coordinates
float * Jz_r, * Jy_r, * Jx_r;                     // random interaction constants
float * Dz_r, * Dy_r, * Dx_r;                     // random interaction constants
float * Jz_b, * Jy_b, * Jx_b;                     // random interaction constants
float * Dz_b, * Dy_b, * Dx_b;                     // random interaction constants

float * red_x, * red_y, * red_z;                  // Auxiliar Matrixes
float * black_x, * black_y, * black_z;            // Auxiliar Matrixes

// counter matrixes
int * cont1;
int * cont2;

float * sum_array_1, * sum_array_2, * sum_array_3;

static int allocate_data (void)
{
  oc = (int *) calloc (N, sizeof(int));
  macr_r = (float *) calloc (RBN, sizeof(float));
  macr_b = (float *) calloc (RBN, sizeof(float));
  mafe_r = (float *) calloc (RBN, sizeof(float));
  mafe_b = (float *) calloc (RBN, sizeof(float));
  mc_r = (float *) calloc (RBN, sizeof(float));
  mc_b = (float *) calloc (RBN, sizeof(float));
  KA_r = (float *) calloc (RBN, sizeof(float));
  KA_b = (float *) calloc (RBN, sizeof(float));

  mx_r = (float *) calloc (RBN, sizeof(float));
  my_r = (float *) calloc (RBN, sizeof(float));
  mz_r = (float *) calloc (RBN, sizeof(float));

  mx_b = (float *) calloc (RBN, sizeof(float));
  my_b = (float *) calloc (RBN, sizeof(float));
  mz_b = (float *) calloc (RBN, sizeof(float));

  Jz_r = (float *) calloc (RBN, sizeof(float));
  Jy_r = (float *) calloc (RBN, sizeof(float));
  Jx_r = (float *) calloc (RBN, sizeof(float));
  Dz_r = (float *) calloc (RBN, sizeof(float));
  Dy_r = (float *) calloc (RBN, sizeof(float));
  Dx_r = (float *) calloc (RBN, sizeof(float));

  Jz_b = (float *) calloc (RBN, sizeof(float));
  Jy_b = (float *) calloc (RBN, sizeof(float));
  Jx_b = (float *) calloc (RBN, sizeof(float));
  Dz_b = (float *) calloc (RBN, sizeof(float));
  Dy_b = (float *) calloc (RBN, sizeof(float));
  Dx_b = (float *) calloc (RBN, sizeof(float));

  red_x = (float *) calloc (RBN, sizeof(float));
  red_y = (float *) calloc (RBN, sizeof(float));
  red_z = (float *) calloc (RBN, sizeof(float));
  
  black_x = (float *) calloc (RBN, sizeof(float));
  black_y = (float *) calloc (RBN, sizeof(float));
  black_z = (float *) calloc (RBN, sizeof(float));

  cont1 = (int *) calloc (RBN, sizeof(int));
  cont2 = (int *) calloc (RBN, sizeof(int));

  sum_array_1 = (float *) calloc (4*L*L, sizeof(float));
  sum_array_2 = (float *) calloc (4*L*L, sizeof(float));
  sum_array_3 = (float *) calloc (4*L*L, sizeof(float));

  std::cout << "size: " << N << "\n"; 
  if (!macr_r || !macr_b || !mafe_r || !mafe_b ||
      !mc_r || !mc_b || !KA_r || !KA_b ||
      !mx_r || !my_r || !mz_r || !mx_b || !my_b || !mz_b ||
      !Jx_r || !Jy_r || !Jz_r || !Jx_b || !Jy_b || !Jz_b ||
      !Dx_r || !Dy_r || !Dz_r || !Dx_b || !Dy_b || !Dz_b ||
      !red_x || !red_y || !red_z || !black_x || !black_y || !black_z ||
      !cont1 || !cont2 || !sum_array_1 || !sum_array_2 || !sum_array_3
      ){
    fprintf(stderr, "cannot allocate data\n");
    exit(0);
  }
  return 1;
}

static void free_data(){
  std::cout << "oc.. " << std::flush;
  if (oc) free(oc);
  std::cout << "macr.. " << std::flush;
  if (macr_r) free(macr_r);
  if (macr_b) free(macr_b);
  std::cout << "mafe_r.. "<< std::flush;
  if (mafe_r) free(mafe_r);
  std::cout << "mafe_b.. "<< std::flush;
  if (mafe_b) free(mafe_b);
  std::cout << "mc.. "<< std::flush;
  if (mc_r) free(mc_r);
  if (mc_b) free(mc_b);
  std::cout << "KA.. "<< std::flush;
  if (KA_r) free(KA_r);
  if (KA_b) free(KA_b);
  

  std::cout << "mx.. "<< std::flush;
  if (mx_r) free(mx_r);
  if (mx_b) free(mx_b);
  std::cout << "my.. "<< std::flush;
  if (my_r) free(my_r);
  if (my_b) free(my_b);
  std::cout << "mz.. "<< std::flush;
  if (mz_r) free(mz_r);
  if (mz_b) free(mz_b);

  std::cout << "Jx.. "<< std::flush;
  if (Jx_r) free(Jx_r);
  if (Jx_b) free(Jx_b);
  std::cout << "Jy.. "<< std::flush;
  if (Jy_r) free(Jy_r);
  if (Jy_b) free(Jy_b);
  std::cout << "Jz.. "<< std::flush;
  if (Jz_r) free(Jz_r);
  if (Jz_b) free(Jz_b);
  std::cout << "Dy.. "<< std::flush;
  if (Dx_r) free(Dx_r);
  if (Dx_b) free(Dx_b);
  std::cout << "Dy.. "<< std::flush;
  if (Dy_r) free(Dy_r);
  if (Dy_b) free(Dy_b);
  std::cout << "Dz.. "<< std::flush;
  if (Dz_r) free(Dz_r);
  if (Dz_b) free(Dz_b);

  std::cout << "red_x.. "<< std::flush;
  if (red_x) free(red_x);
  std::cout << "red_y.. "<< std::flush;
  if (red_y) free(red_y);
  std::cout << "red_z.. "<< std::flush;
  if (red_z) free(red_z);

  std::cout << "black_x.. "<< std::flush;
  if (black_x) free(black_x);
  std::cout << "black_y.. "<< std::flush;
  if (black_y) free(black_y);
  std::cout << "black_z.. "<< std::flush;
  if (black_z) free(black_z);


  if (cont1) free(cont1);
  if (cont2) free(cont2);

  if (sum_array_1) free(sum_array_1);
  if (sum_array_2) free(sum_array_2);
  if (sum_array_3) free(sum_array_3);
};

/*
This function defines the element for each site of the cubic matrix
*/
void set_interac(trng::lcg64_shift * Rng, trng::uniform_dist<float> Uni){

    init_oc(oc, macr_r, mafe_r, L, Rng, Uni, RED_TILES);

    init_oc(oc, macr_b, mafe_b, L, Rng, Uni, BLACK_TILES);

    interac_init(oc, KA_r, Jx_r, Jy_r, Jz_r,
                 Dx_r, Dy_r, Dz_r, RED_TILES, L);

    interac_init(oc, KA_b, Jx_b, Jy_b, Jz_b,
                 Dx_b, Dy_b, Dz_b, BLACK_TILES, L);

}

void inline swap(float* &a, float* &b) {
    float * temp = a;
    a = b;
    b = temp;
}


int main(int argc, char * argv[]){
    bool save_files = false;
    bool graphics = false;
    bool size = false;
    bool temps = false;
    bool fracs = false;

    while (*++argv){
      if ((*argv)[0] == '-'){
        switch ((*argv)[1]){
          case 'h':
              std::cout << "\n\t-g for graphic.\n\t-o for file output.";
              std::cout << "\n\t-L X where X is an integer for the size";
              std::cout << "considering N=(2*L)^3";
              std::cout << "\n\t-t for the t_mic tmax parameters";
			  std::cout << "\n\t-f for the frac and flucfrac parameters";
              exit(0);
          case 'o':
              save_files = true;
              std::cout << "\n\tFile output enables\n";
              break;
          case 'g':
              graphics = true;
              std::cout << "\n\tGraphics representation activated\n";
              break;
          case 'L':
              L = atoi(argv[1]);
              N = 8*L*L*L;
              RBN = 4*L*L*L;
              size = true;
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
              fracs = true;
              break;
          case 'm':
              MAX_T = atoi(argv[1]);
              std::cout << "\n\tMax number of threads: " << MAX_T;
              break;
          default:
              std::cout << "\n\tUnknown option -\n\n" << (*argv)[1] << "\n";
              break;
        }
      }
    }

    if (!size){
      std::cout << "Please run again and add -L X to the command line ";
      std::cout << " where X is an integer for the size, considering ";
      std::cout << "N = (2*L)^3.\n";
      exit(1);
    }
    if (!temps){
        t_mic = 100;
        tmax = 1000;
    }
    if (!fracs){
        frac = 0.55f;
        flucfrac = 0.35f;
    }


    int i,j,k,t,ll1,ll3;
    float R, tmp1, tmp2;
    float s_mx, s_my, s_mz;
    float s_mx_1, s_my_1, s_mz_1;
    float s_mx_2, s_my_2, s_mz_2;

    float cr_mx, cr_my, cr_mz, cr_mx_1, cr_my_1, cr_mz_1, cr_mx_2, cr_my_2;
    float cr_mz_2;
    float fe_mx, fe_my, fe_mz, fe_mx_1, fe_my_1, fe_mz_1, fe_mx_2, fe_my_2;
    float fe_mz_2;
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

    // accepted changes vars
    int suma = 0;
    int counter = 0;

    // auxiliars for loops neighbours indexes
    int cjm = 0;
    int ckm = 0;
    int cim = 0;
    int index = 0;
    int idx_jk, idx_ik, idx_ij;
    int idx_ijk;
    int idx_k, idx_j, idx_i;

    trng::lcg64_shift Rng[MAX_T];
    trng::uniform_dist<float> Uni(0,1);
    for (int i=0; i<MAX_T; i++){
      Rng[i].split(MAX_T,i);
    }

    allocate_data();

    // initialization of matrixs
    ll1 = 0;
    ll3 = 0;
    std::cout << "about to save conf\n";
    set_interac(Rng, Uni);


    std::cout << ". saved" << "\n";

    //fills mx according to oc values
    mc_init(oc, mc_r, L, RED_TILES);
    mc_init(oc, mc_b, L, BLACK_TILES);

    MyApp app;
    if (graphics){
      app.App_init();
      app.init(L);
    }

    std::cout << "\n";
    std::cout << "initialized mc" << "\n";

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

    total_time = omp_get_wtime();

    for (ll3=0; ll3 < Nsamp; ll3++){

      Temp = Tempmax;
      //initial random spin configuration.
      init_fill(mx_r, my_r, mz_r, RED_TILES, L, Rng, Uni);
      init_fill(mx_b, my_b, mz_b, BLACK_TILES, L, Rng, Uni);

      std::cout << "initialized mx my mz" << "\n";

      H = Hfc;

      for (ll1=0; ll1<Nptos; ll1++){

        std::cout << " round: " << ll1 << "  - " <<std::flush;
        Temp -= (Tempmax-Tempmin)/Nptos;

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
        // Termalizacion previa
        for (t=0; t<tmax; t++){

            memset(cont1, 0, RBN*sizeof(int));
            memset(cont2, 0, RBN*sizeof(int));
            start = omp_get_wtime();
            update(mx_r, my_r, mz_r, KA_r, mc_r, Jx_r, Jy_r, Jz_r, Dx_r,
                   Dy_r, Dz_r, Jx_b, Jy_b, Jz_b, Dx_b, Dy_b, Dz_b,
                   RED_TILES, red_x, red_y, red_z, mx_b, my_b,
                   mz_b, L, Hx, Hy, Hz, Temp, cont1, Rng, Uni);

            update(mx_b, my_b, mz_b, KA_b, mc_b, Jx_b, Jy_b, Jz_b, Dx_b,
                   Dy_b, Dz_b, Jx_r, Jy_r, Jz_r, Dx_r, Dy_r, Dz_r,
                   BLACK_TILES, black_x, black_y, black_z, mx_r, my_r,
                   mz_r, L, Hx, Hy, Hz, Temp, cont2, Rng, Uni);

            stop = omp_get_wtime();

            #pragma omp parallel for shared (cont1, cont2) reduction(+: suma)
            for (int i=0; i<RBN; i++){
              suma += cont1[i] + cont2[i];
            }


            swap(mx_r,red_x);
            swap(my_r,red_y);
            swap(mz_r,red_z);

            swap(mx_b,black_x);
            swap(my_b,black_y);
            swap(mz_b,black_z);

            aux_termal += 1.0e9*(stop- start) / (N);
            counter +=1;
            if (graphics)
              app.drawOne(mx_r, my_r, mz_r, mx_b, my_b, mz_b);
        }

        std::cout << " average: " << (suma/(float)counter)/(float)N;
        suma = 0;
        counter = 0;

        for (t=0; t<tmax; t++){

            memset(cont1, 0, RBN*sizeof(int));
            memset(cont2, 0, RBN*sizeof(int));
            start = omp_get_wtime();
            update(mx_r, my_r, mz_r, KA_r, mc_r, Jx_r, Jy_r, Jz_r, Dx_r,
                   Dy_r, Dz_r, Jx_b, Jy_b, Jz_b, Dx_b, Dy_b, Dz_b,
                   RED_TILES, red_x, red_y, red_z, mx_b, my_b,
                   mz_b, L, Hx, Hy, Hz, Temp, cont1, Rng, Uni);

            update(mx_b, my_b, mz_b, KA_b, mc_b, Jx_b, Jy_b, Jz_b, Dx_b,
                   Dy_b, Dz_b, Jx_r, Jy_r, Jz_r, Dx_r, Dy_r, Dz_r,
                   BLACK_TILES, black_x, black_y, black_z, mx_r, my_r,
                   mz_r, L, Hx, Hy, Hz, Temp, cont2, Rng, Uni);

            stop = omp_get_wtime();

            #pragma omp parallel for shared(cont1, cont2) reduction(+: suma)
            for (int i=0; i<RBN; i++){
              suma += cont1[i] + cont2[i];
            }

            swap(mx_r,red_x);
            swap(my_r,red_y);
            swap(mz_r,red_z);

            swap(mx_b,black_x);
            swap(my_b,black_y);
            swap(mz_b,black_z);

            counter += 1;
            aux_second += 1.0e9*(stop- start) / (N);
            if (graphics)
                            app.drawOne(mx_r, my_r, mz_r, mx_b, my_b, mz_b);

            R += suma;

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


              calculate(mx_r, my_r, mz_r, mc_r, sum_array_1, sum_array_2, sum_array_3,
                        RED_TILES, L);
              #pragma omp parallel for shared (sum_array_1,sum_array_2,sum_array_3)\
              reduction(+:tmp1_x, tmp1_y, tmp1_z)
              for (int i=0; i<4*L*L; i++){
                tmp1_x += sum_array_1[i];
                tmp1_y += sum_array_2[i];
                tmp1_z += sum_array_3[i];
              }

              calculate(mx_b, my_b, mz_b, mc_b, sum_array_1, sum_array_2, sum_array_3,
                        BLACK_TILES, L);
              #pragma omp parallel for shared (sum_array_1,sum_array_2,sum_array_3)\
              reduction(+:tmp2_x, tmp2_y, tmp2_z)
              for (int i=0; i<4*L*L; i++){
                tmp2_x += sum_array_1[i];
                tmp2_y += sum_array_2[i];
                tmp2_z += sum_array_3[i];
              }

              calculate(mx_r, my_r, mz_r, macr_r, sum_array_1, sum_array_2, sum_array_3,
                        RED_TILES, L);
              #pragma omp parallel for shared (sum_array_1,sum_array_2,sum_array_3)\
              reduction(+:cr1_x, cr1_y, cr1_z)
              for (int i=0; i<4*L*L; i++){
                cr1_x += sum_array_1[i];
                cr1_y += sum_array_2[i];
                cr1_z += sum_array_3[i];
              }

              calculate(mx_b, my_b, mz_b, macr_b, sum_array_1, sum_array_2, sum_array_3,
                        BLACK_TILES, L);
              #pragma omp parallel for shared(sum_array_1,sum_array_2,sum_array_3)\
              reduction(+:cr2_x, cr2_y, cr2_z)
              for (int i=0; i<4*L*L; i++){
                cr2_x += sum_array_1[i];
                cr2_y += sum_array_2[i];
                cr2_z += sum_array_3[i];
              }

              calculate(mx_r, my_r, mz_r, mafe_r, sum_array_1, sum_array_2, sum_array_3,
                        RED_TILES, L);
              #pragma omp parallel for shared(sum_array_1,sum_array_2,sum_array_3)\
              reduction(+:fe1_x, fe1_y, fe1_z)
              for (int i=0; i<4*L*L; i++){
                fe1_x += sum_array_1[i];
                fe1_y += sum_array_2[i];
                fe1_z += sum_array_3[i];
              }

              calculate(mx_b, my_b, mz_b, mafe_b, sum_array_1, sum_array_2, sum_array_3,
                        BLACK_TILES, L);
              #pragma omp parallel for shared(sum_array_1,sum_array_2,sum_array_3)\
              reduction(+:fe2_x, fe2_y, fe2_z)
              for (int i=0; i<4*L*L; i++){
                fe2_x += sum_array_1[i];
                fe2_y += sum_array_2[i];
                fe2_z += sum_array_3[i];
              }

              fe_mx += mfe*(fe1_x+fe2_x)/N;
              fe_my += mfe*(fe1_y+fe2_y)/N;
              fe_mz += mfe*(fe1_z+fe2_z)/N;

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

              mag += (((tmp1_x+tmp2_x)/N)*Hx+((tmp1_y+tmp2_y)/N)*Hy+((tmp1_z+tmp2_z)/N)*Hz)/H;
              m1  += 2.0f*sqrt(p(tmp1_x/N,2)+p(tmp1_y/N,2)+p(tmp1_z/N,2));
              m2  += 4.0f*(p(tmp1_x/N,2)+p(tmp1_y/N,2)+p(tmp1_z/N,2));
              m4  += 16.0f*p(p(tmp1_x/N,2)+p(tmp1_y/N,2)+p(tmp1_z/N,2),2);

              E_afm_ex = 0.0f;
              E_afm_an = 0.0f;
              E_afm_dm = 0.0f;

              energy(mx_r, my_r, mz_r, sum_array_1, sum_array_2, sum_array_3, KA_r, mc_r,
                     Jz_r, Jy_r, Jx_r, Dz_r, Dy_r, Dx_r,
                     Jz_b, Jy_b, Jx_b, Dz_b, Dy_b, Dx_b,
                     mx_b, my_b, mz_b, RED_TILES, L);

              #pragma omp parallel for shared(sum_array_1,sum_array_2,sum_array_3)\
              reduction(+:E_afm_ex, E_afm_an, E_afm_dm)
              for (int i=0; i<4*L*L; i++){
                E_afm_an += sum_array_1[i];
                E_afm_ex += sum_array_2[i];
                E_afm_dm += sum_array_3[i];
              }

              energy(mx_b, my_b, mz_b, sum_array_1, sum_array_2, sum_array_3, KA_b, mc_b,
                     Jz_b, Jy_b, Jx_b, Dz_b, Dy_b, Dx_b,
                     Jz_r, Jy_r, Jx_r, Dz_r, Dy_r, Dx_r,
                     mx_r, my_r, mz_r, BLACK_TILES, L);
              #pragma omp parallel for shared(sum_array_1,sum_array_2,sum_array_3)\
              reduction(+:E_afm_ex, E_afm_an, E_afm_dm)
              for (int i=0; i<4*L*L; i++){
                E_afm_an += sum_array_1[i];
                E_afm_ex += sum_array_2[i];
                E_afm_dm += sum_array_3[i];
              }

              E_afm_ex = 0.5f*E_afm_ex/N;  //la energia de intercambio se cuenta dos veces
              E_afm_dm = 0.5f*E_afm_dm/N;  //la energia de intercambio se cuenta dos veces
              E_afm_an = E_afm_an/N;

              U += E_afm_ex + E_afm_an + E_afm_dm;
              U2 += p(E_afm_ex + E_afm_an + E_afm_dm,2);

            }
        }

        suma = 0;
        counter = 0;


        nseconds = (aux_termal + aux_second)/(tmax*2); 
        high = high > nseconds ? high : nseconds ;
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

        ss_m_2[ll1] += 2.0f*sqrt(p(s_mx_2,2) + p(s_my_2,2) + p(s_mz_2,2))
                                 *t_mic/tmax;
        ss_m_1[ll1] += m1*t_mic/tmax;

        std::cout << " ss_m_1["<< ll1<< "]:" << ss_m_1[ll1] << "\n";

        ss_m2[ll1] += m2*t_mic/tmax;
        ss_m4[ll1] += m4*t_mic/tmax; // Binder's calc

        ss_m[ll1] += mag*t_mic/tmax;
        ss_U[ll1] += U*t_mic/tmax;
        ss_U2[ll1] += U2*t_mic/tmax;

        R_a[ll1] += R/tmax*N;      // acceptance ratio

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
      // output files
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

      std::cout << " about to loop write" << "\n";
      for (i=0; i<Nptos; i++){

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

  std::cout << "about to free" << "\n";
  free_data();

  return 0;

}
