#pragma once
#include <stdio.h> // For `printf()`.
#include <stdexcept> // For std::runtime_error().
#include <cuda_runtime.h>

void _cuda_assert(cudaError_t error, const char* call_file, unsigned int call_line, const char* expression) {
  if (error==cudaSuccess) return;
  printf("Assertion in %s:%d %s. %s\n", call_file, call_line, expression, cudaGetErrorString(error));
  throw std::runtime_error(cudaGetErrorString(error));
}
#define cuda_assert(expr) _cuda_assert(expr, __FILE__, __LINE__, #expr);
