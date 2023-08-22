#pragma once

#include <cuda_runtime.h> // For cuda*.
#include <cudart.hpp> // For cuda_assert.

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

#include <cuDNN.hpp> // For cuDNN::*.

typedef cuDNN::Format Format;

template <typename T>
class Tensor: public cuDNN::TensorDescriptor, public DeviceData {
public:
  void Create(int batch_size, int depth, int height, int width, const void* data = nullptr, Format format = Format::NCHW) {
    cuDNN::TensorDescriptor::Create<T>(batch_size, depth, height, width, (cudnnTensorFormat_t)format);
    DeviceData::Create(batch_size * depth * height * width * sizeof(T), data);
  }
  void Destroy() {
    DeviceData::Destroy();
    cuDNN::TensorDescriptor::Destroy();
  }
};

template <typename T>
class Filter: public cuDNN::FilterDescriptor, public DeviceData {
public:
  void Create(int output_depth, int input_depth, int height, int width, const void* data = nullptr, Format format = Format::NCHW) {
    cuDNN::FilterDescriptor::Create<T>(output_depth, input_depth, height, width, (cudnnTensorFormat_t)format);
    DeviceData::Create(output_depth * input_depth * height * width * sizeof(T), data);
  }
  void Destroy() {
    DeviceData::Destroy();
    cuDNN::FilterDescriptor::Destroy();
  }
};

template <typename T>
class CrossCorrelation: public cuDNN::ConvolutionDescriptor {
  cuDNN::Handle handle;
  cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t workspace_size = 0;
  void* workspace_data_device = nullptr;
public:
  void Create() {
    cuDNN::ConvolutionDescriptor::Create<T>(CUDNN_CROSS_CORRELATION);
    handle.Create();
  }
  template <typename T2>
  void Configure(const Tensor<T2>& input, const Filter<T2>& filter, const Tensor<T2>& output) {
    convolution_algorithm = FindAlgorithm(handle, input, filter, output);

    workspace_size = WorkspaceSize(handle, input, filter, output, convolution_algorithm);
    cuda_assert(cudaMalloc(&workspace_data_device, workspace_size));
  }
  template <typename T2>
  void* Run(const Tensor<T2>& input, const Filter<T2>& filter, Tensor<T2>& output) {
    Forward(
      handle,
      input, input.Data(),
      filter, filter.Data(),
      *this, convolution_algorithm, workspace_data_device, workspace_size,
      output, output.Data()
    );
    cudaDeviceSynchronize();
    return output.Data();
  }
  void Destroy() {
    if (workspace_data_device) cuda_assert(cudaFree(workspace_data_device));
    handle.Destroy();
    cuDNN::ConvolutionDescriptor::Destroy();
  }
};
