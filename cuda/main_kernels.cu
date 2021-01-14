#include "cub/cub.cuh"
#include "variables.h"
#include "main_kernels.h"

__global__
void add_one_var(const float * const __restrict__ mx, const float * const __restrict__ my,
                 const float * const __restrict__ mz,
                 const float * const __restrict__ elem,
                 float * __restrict__ am_x, float * __restrict__ am_y, float * __restrict__ am_z,
                 const int L, const grid_color color) {

    int x, y, z;
    x = threadIdx.x + blockDim.x*blockIdx.x;
    y = threadIdx.y + blockDim.y*blockIdx.y;
    z = threadIdx.z + blockDim.z*blockIdx.z;

    if (x < L && y < 2*L && z < 2*L) {

        float var_x = 0.0f;
        float var_y = 0.0f;
        float var_z = 0.0f;

        int rb_1 = 0, rb_2 = 0, rb_3 = 0;
        int ix = 4*x;
        int idx_rb = RB(ix, y, z);        // index for color matrix

        // this indexs are the neighbours in the other color matrix
        rb_1 = RB(ix+1, y, z);
        rb_2 = RB(ix+2, y, z);
        rb_3 = RB(ix+3, y, z);

        var_x = mx[rb_1]*elem[rb_1] + mx[rb_2]*elem[rb_2];
        var_y = my[rb_1]*elem[rb_1] + my[rb_2]*elem[rb_2];
        var_z = mz[rb_1]*elem[rb_1] + mz[rb_2]*elem[rb_2];

        var_x += mx[rb_3]*elem[rb_3] + mx[idx_rb]*elem[idx_rb];
        var_y += my[rb_3]*elem[rb_3] + my[idx_rb]*elem[idx_rb];
        var_z += mz[rb_3]*elem[rb_3] + mz[idx_rb]*elem[idx_rb];

        am_x[idx_rb] = var_x;
        am_y[idx_rb] = var_y;
        am_z[idx_rb] = var_z;
    }

}

__host__
void calculate(float * __restrict__ var1_x, float * __restrict__ var1_y, float * __restrict__ var1_z,
               float * __restrict__ var2_x, float * __restrict__ var2_y, float * __restrict__ var2_z,
               const float * const __restrict__ mx_r, const float * const __restrict__ my_r,
               const float * const __restrict__ mz_r,
               const float * const __restrict__ mx_b, const float * const __restrict__ my_b,
               const float * const __restrict__ mz_b,
               const float * const __restrict__ elem_r, const float * const __restrict__ elem_b,
               float * am1_x, float * am1_y, float * am1_z,
               float * am2_x, float * am2_y, float * am2_z,
               float * out, void * d_ts, size_t d_ts_bytes,
               const int L, const dim3 blocks, const dim3 threads) {

    const int RBN = 4*L*L*L;
    dim3 d_t = dim3((threads.x+4-1)/4, threads.y, threads.z);

    // this are Auxiliar Matrixes (am)
    cudaMemset(am1_x, 0.0f, RBN*sizeof(float));
    cudaMemset(am1_y, 0.0f, RBN*sizeof(float));
    cudaMemset(am1_z, 0.0f, RBN*sizeof(float));
    cudaMemset(am2_x, 0.0f, RBN*sizeof(float));
    cudaMemset(am2_y, 0.0f, RBN*sizeof(float));
    cudaMemset(am2_z, 0.0f, RBN*sizeof(float));

    add_one_var<<<blocks, d_t>>>(mx_r, my_r, mz_r, elem_r,
                                 am1_x, am1_y, am1_z, L, RED_TILES);
    getLastCudaError("temp red kernel failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    add_one_var<<<blocks, d_t>>>(mx_b, my_b, mz_b, elem_b,
                                 am2_x, am2_y, am2_z, L, BLACK_TILES);
    getLastCudaError("temp black kernel failed\n");
    checkCudaErrors(cudaDeviceSynchronize());


    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am1_x, out, RBN));
    checkCudaErrors(cudaDeviceSynchronize());
    *var1_x = out[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am1_y, out, RBN));
    checkCudaErrors(cudaDeviceSynchronize());
    *var1_y = out[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am1_z, out, RBN));
    checkCudaErrors(cudaDeviceSynchronize());
    *var1_z = out[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am2_x, out, RBN));
    checkCudaErrors(cudaDeviceSynchronize());
    *var2_x = out[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am2_y, out, RBN));
    checkCudaErrors(cudaDeviceSynchronize());
    *var2_y = out[0];

    CubDebugExit(cub::DeviceReduce::Sum(d_ts, d_ts_bytes, am2_z, out, RBN));
    checkCudaErrors(cudaDeviceSynchronize());
    *var2_z = out[0];
}
