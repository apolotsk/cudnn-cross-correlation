#pragma once

#include <stdio.h> // For `printf()`.
#include <stdlib.h> // For `exit()`.
#include <cuda_runtime.h>
void _cuda_assert(cudaError_t error, int line) {
  if (error==cudaSuccess) return;
  printf("Error at line %d: %s.\n", line, cudaGetErrorString(error));
  exit(1);
}
#define cuda_assert(status) _cuda_assert(status, __LINE__);
