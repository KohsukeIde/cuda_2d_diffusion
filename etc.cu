#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "func.h"

void swap_variables(float **dependent_variable, float **updated_variable) {
  float *temp = *dependent_variable;
  *dependent_variable = *updated_variable;
  *updated_variable = temp;
}

void allocate_memory(int num_gpus, int x_size, int y_size, float **dependent_variable, float **updated_variable) {
  int size = x_size * y_size;
  if (num_gpus > 0) {
    cudaError_t error = cudaMalloc((void**)dependent_variable, x_size * y_size * sizeof(float));
    if (error != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void**)updated_variable, x_size * y_size * sizeof(float));
    if (error != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }
  } else {
    *dependent_variable = (float*)malloc(sizeof(float) * size);
    *updated_variable = (float*)malloc(sizeof(float) * size);
  }
}

void generate_bmp_image(int num_gpus, int x_size, int y_size, float *dependent_variable, int multiple_scale, float max_value, float min_value, char *filename, char *palette) {
  int j, jx, jy, ix, iy, jmx, jmy, mul_x, mul_y;
  float gux, guy, fc, gmax = -1.0e+30, gmin = 1.0e+30, f0, f1, *data;
  static int mx = 0, my = 0;
  char *image_data;

  mx = multiple_scale * (x_size - 1) + 1;
  my = multiple_scale * (y_size - 1) + 1;
  image_data = (char*)malloc(sizeof(float) * mx * my);

  if (num_gpus > 0) {
    data = (float*)malloc(x_size * y_size * sizeof(float));
    cudaMemcpy(data, dependent_variable, x_size * y_size * sizeof(float), cudaMemcpyDeviceToHost);
  } else {
    data = dependent_variable;
  }

  for (jy = 0; jy < y_size - 1; jy++) {
    for (jx = 0; jx < x_size - 1; jx++) {
      j = x_size * jy + jx;

      if (jx == x_size - 1) mul_x = 1;
      else mul_x = multiple_scale;
      if (jy == y_size - 1) mul_y = 1;
      else mul_y = multiple_scale;
      for (jmy = 0; jmy < mul_y; jmy++) {
        iy = multiple_scale * jy + jmy;
        iy = my - 1 - iy;
        for (jmx = 0; jmx < mul_x; jmx++) {
          ix = multiple_scale * jx + jmx;
          gux = (float)jmx / (float)mul_x;
          guy = (float)jmy / (float)mul_y;

          f0 = (1.0 - gux) * data[j] + gux * data[j + 1];
          f1 = (1.0 - gux) * data[j + x_size] + gux * data[j + x_size + 1];
          fc = (1.0 - guy) * f0 + guy * f1;

          gmax = fmax(gmax, fc);
          gmin = fmin(gmin, fc);
          fc = 253.0 * (fc - min_value) / (max_value - min_value) + 2.0;

          image_data[mx * iy + ix] = (char)fmin(fmax(2.0, fc), 253.0);
        }
      }
    }
  }

  DFR8bmp(image_data, mx, my, filename, palette);
  fprintf(stderr,"filename=%s  ",filename);
  fprintf(stderr,"MAX=%10.3e  MIN=%10.3e\n",gmax, gmin);

  free(image_data);
  if(num_gpus > 0) free(data);

  return;
}


int set_num_gpus(int numGPUs, int argc, char **argv) {
    int usrGPUs;
    if (argc > 1) {
        if (strncmp(argv[1], "-gpu", 4) == 0) {
            usrGPUs = numGPUs;
        } else if (strncmp(argv[1], "-cpu", 4) == 0) {
            usrGPUs = 0;
        } else {
            usrGPUs = numGPUs;
        }
    } else {
        usrGPUs = numGPUs;
    }
    return usrGPUs;
}
