#define  NX  256
#define  NY  256

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "func.h"

int main(int argc, char *argv[])
{
  int numGPUs;
  cudaGetDeviceCount(&numGPUs);
  numGPUs = set_num_gpus(numGPUs,argc,argv);

  int nx = NX, ny = NY, icnt = 1, nout = 600;
  float Lx = 1.0, Ly = 1.0;
  float dx = Lx/(float)nx, dy = Ly/(float)ny, kappa = 0.1;
  float *f, *fn, dt, time = 0.0, flops = 0.0;
  char filename[] = "f000.bmp";

  allocate_memory(numGPUs, nx, ny, &f, &fn);
  initial(numGPUs, nx, ny, dx, dy, f);
  dt = (0.20*MIN(dx*dx, dy*dy)/kappa);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  do {
    if (icnt % 100 == 0) {
      fprintf(stderr,"time(%4d)=%7.5f\n", icnt, time + dt);
    }
    flops += diffusion_2d(numGPUs, nx, ny, f, fn, kappa, dt, dx, dy);
    swap_variables(&f, &fn);
    time += dt;
    if (icnt % nout == 0) {
      printf("TIME = %9.3e\n", time);
      sprintf(filename, "f%03d.bmp", icnt/nout);
      generate_bmp_image(numGPUs, nx, ny, f, 1, 1.11, -1.0, filename, "");
    }
  } while (icnt++ < 999999 && time + 0.5*dt < 0.9);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed Time= %9.3e [ms]\n", milliseconds);

  return 0;
}

void initial(int numGPUs, int nx, int ny, float dx, float dy, float *f)
{
  int j, jx, jy;
  float *F, x, y, alpha = 30.0;

  if (numGPUs > 0) {
    F = (float *) malloc(nx*ny*sizeof(float));
  }
  else {
    F = f;
  }

  for (jy = 0; jy < ny; jy++) {
    for (jx = 0; jx < nx; jx++) {
      j = nx*jy + jx;
      x = dx*((float)jx + 0.5) - 0.5;
      y = dy*((float)jy + 0.5) - 0.5;
      F[j] = exp(-alpha*(x*x + y*y));
    }
  }

  if (numGPUs > 0) {
    cudaError_t err(cudaMemcpy(f, F, nx*ny*sizeof(float), cudaMemcpyHostToDevice));
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy(f) failed: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
    free(F);
  }
}
