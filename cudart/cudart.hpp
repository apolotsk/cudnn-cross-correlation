#pragma once
#include <cstdio> // For printf.
#include <stdexcept> // For std::runtime_error.
#include <cuda_runtime.h> // For cuda*.

/** \brief <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/index.html">CUDA Runtime API</a> for C++. */
namespace cudart {

/**
 * \brief Assert a cuda runtime function call.
 * 
 * If there is an error, then throw a runtime error.
 *
 * \param expression A cuda runtime function call. For example, `cudaMalloc(...)`.
 */
#define cuda_assert(expression) _cuda_assert(expression, __FILE__, __LINE__, #expression);
/** \brief Is intended to be called by the macro `cuda_assert` only. */
void _cuda_assert(cudaError_t error, const char* call_file, unsigned int call_line, const char* expression) {
  if (error==cudaSuccess) return;
  printf("Assertion in %s:%d %s. %s\n", call_file, call_line, expression, cudaGetErrorString(error));
  throw std::runtime_error(cudaGetErrorString(error));
}

/**
 * \brief Data in the device memory.
 * 
 * A class for managing data allocation of copying in the device (CUDA) memory.
 */
class DeviceData {
  /** \brief Data size in bytes. */
  size_t size;
  /** \brief Data pointer in the device memory. */
  void* data = nullptr;
public:
  /**
   * \brief Create instance.
   * \param size Data size in bytes.
   * \param data Data pointer in the system memory. If specified, it will be copied to the device memory.
   */
  void Create(size_t size, const void* data = nullptr) {
    this->size = size;
    cuda_assert(cudaMalloc(&this->data, size));
    if (data) CopyFrom(data);
  }
  /** \brief Get data size in bytes. */
  size_t Size() const { return size; }
  /** \brief Get data pointer in the device memory. */
  void* Data() { return data; }
  /** \brief Get data pointer in the device memory. */
  const void* Data() const { return data; }

  /** \brief Copy `data` from the system memory. */
  void CopyFrom(const void* data) {
    cuda_assert(cudaMemcpy(this->data, data, size, cudaMemcpyHostToDevice));
  }
  /** \brief Copy `data` to the system memory. */
  void* CopyTo(void* data) const {
    cuda_assert(cudaMemcpy(data, this->data, size, cudaMemcpyDeviceToHost));
    return data;
  }
  /** \brief Destroy instance. */
  void Destroy() {
    if (data) cuda_assert(cudaFree(data));
  }
};

}
