#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "func.h"

#define BLOCK_DIM_X 256
#define BLOCK_DIM_Y 8

__global__ void diffusion_2d_cuda(float *dependent_var, float *updated_dependent_var,
                                  int grid_size_x, int grid_size_y, float coef_0, float coef_1, float coef_2)
{
    int j, js = threadIdx.x + 1, jy, j0 = 0, j1 = 1, j2 = 2, tmp;
    __shared__ float shared_fs[3][BLOCK_DIM_X + 2];

    jy = BLOCK_DIM_Y * blockIdx.y;
    j = grid_size_x * jy + blockDim.x * blockIdx.x + threadIdx.x;
    shared_fs[j1][js] = dependent_var[j];

    if (blockIdx.y == 0)
        shared_fs[j0][js] = shared_fs[j1][js];
    else
        shared_fs[j0][js] = dependent_var[j - grid_size_x];
    j += grid_size_x;

#pragma unroll
    for (jy = 0; jy < BLOCK_DIM_Y; jy++)
    {
        if (blockIdx.y == gridDim.y - 1)
            shared_fs[j2][js] = shared_fs[j1][js];
        else
            shared_fs[j2][js] = dependent_var[j];

        if (threadIdx.x == 0)
        {
            if (blockIdx.x == 0)
                shared_fs[j1][0] = shared_fs[j1][1];
            else
                shared_fs[j1][0] = dependent_var[j - grid_size_x - 1];
        }
        if (threadIdx.x == blockDim.x - 1)
        {
            if (blockIdx.x == gridDim.x - 1)
                shared_fs[j1][js + 1] = shared_fs[j1][js];
            else
                shared_fs[j1][js + 1] = dependent_var[j - grid_size_x + 1];
        }

        __syncthreads();

        updated_dependent_var[j - grid_size_x] = coef_0 * (shared_fs[j1][js - 1] + shared_fs[j1][js + 1]) +
                                                 coef_1 * (shared_fs[j0][js] + shared_fs[j2][js]) +
                                                 coef_2 * shared_fs[j1][js];

        j += grid_size_x;

        tmp = j0;
        j0 = j1;
        j1 = j2;
        j2 = tmp;
    }
}

void diffusion_2d_cpu(float *f, float *fn, int nx, int ny, float c0, float c1, float c2) {
    int i, x, y;
    float center, east, west, south, north;

    for (y = 0; y < ny; y++) {
        for (x = 0; x < nx; x++) {
            i = nx*y + x;
            center = f[i];

            if (x == 0) {
                west = center;
            } else {
                west = f[i - 1];
            }

            if (x == nx - 1) {
                east = center;
            } else {
                east = f[i + 1];
            }

            if (y == 0) {
                south = center;
            } else {
                south = f[i - nx];
            }

            if (y == ny - 1) {
                north = center;
            } else {
                north = f[i + nx];
            }

            fn[i] = c0*(east + west) + c1*(north + south) + c2*center;
        }
    }
}

float diffusion_2d(int num_gpus, int grid_size_x, int grid_size_y, float *dependent_var, float *updated_dependent_var,
                  float diff_coef, float time_step, float grid_spacing_x, float grid_spacing_y)
{
    float coef_0 = diff_coef * time_step / (grid_spacing_x * grid_spacing_x);
    float coef_1 = diff_coef * time_step / (grid_spacing_y * grid_spacing_y);
    float coef_2 = 1.0 - 2.0 * (coef_0 + coef_1);

    if (num_gpus > 0)
    {
        dim3 grid(grid_size_x / BLOCK_DIM_X, grid_size_y / BLOCK_DIM_Y, 1);
        dim3 threads(BLOCK_DIM_X, 1, 1);
        diffusion_2d_cuda<<<grid, threads>>>(dependent_var, updated_dependent_var, grid_size_x, grid_size_y, coef_0, coef_1, coef_2);

        cudaDeviceSynchronize();
    }
    else
    {
        diffusion_2d_cpu(dependent_var, updated_dependent_var, grid_size_x, grid_size_y, coef_0, coef_1, coef_2);
    }

    return (float)(grid_size_x * grid_size_y) * 7.0;
}
