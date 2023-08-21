#pragma once
#include <cstdio> // For printf.
#include <stdexcept> // For std::runtime_error.
#include <cuda_runtime.h> // For cuda*.

namespace cudart {

void _cuda_assert(cudaError_t error, const char* call_file, unsigned int call_line, const char* expression) {
  if (error==cudaSuccess) return;
  printf("Assertion in %s:%d %s. %s\n", call_file, call_line, expression, cudaGetErrorString(error));
  throw std::runtime_error(cudaGetErrorString(error));
}
#define cuda_assert(expr) _cuda_assert(expr, __FILE__, __LINE__, #expr);

class DeviceData {
  size_t size;
  void* data = nullptr;
public:
  void Create(size_t size, const void* data = nullptr) {
    this->size = size;
    cuda_assert(cudaMalloc(&this->data, size));
    if (data) CopyFrom(data);
  }
  size_t Size() const { return size; }
  void* Data() { return data; }
  const void* Data() const { return data; }

  void CopyFrom(const void* data) {
    cuda_assert(cudaMemcpy(this->data, data, size, cudaMemcpyHostToDevice));
  }
  void* CopyTo(void* data) const {
    cuda_assert(cudaMemcpy(data, this->data, size, cudaMemcpyDeviceToHost));
    return data;
  }
  void Destroy() {
    if (data) cuda_assert(cudaFree(data));
  }
};

}
