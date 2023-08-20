#pragma once
#include <cuda_runtime.h> // For cuda*.
#include <cudart.hpp> // For cuda_assert.
#include <cudnn.hpp> // For cudnn::*.

typedef __fp16 half;
template <typename T> cudnnDataType_t data_type;
template <> cudnnDataType_t data_type<half> = CUDNN_DATA_HALF;
template <> cudnnDataType_t data_type<float> = CUDNN_DATA_FLOAT;
template <> cudnnDataType_t data_type<double> = CUDNN_DATA_DOUBLE;

enum Format {
  NCHW = CUDNN_TENSOR_NCHW,
  NHWC = CUDNN_TENSOR_NHWC,
};

class Data {
public:
  size_t size;
  void* data = NULL;
  void Create(size_t size, const void* data = NULL) {
    this->size = size;
    cuda_assert(cudaMalloc(&this->data, size));
    if (data) CopyTo(data);
  }
  void CopyTo(const void* data) {
    cuda_assert(cudaMemcpy(this->data, data, size, cudaMemcpyHostToDevice));
  }
  void* CopyFrom(void* data) const {
    cuda_assert(cudaMemcpy(data, this->data, size, cudaMemcpyDeviceToHost));
    return data;
  }
  void Destroy() {
    if (data) cuda_assert(cudaFree(data));
  }
};

template <typename T>
class Tensor: public cudnn::TensorDescriptor, public Data {
public:
  int batch_size, depth, height, width;
  void Create(int batch_size, int depth, int height, int width, const void* data = NULL, Format format = NCHW) {
    cudnn::TensorDescriptor::Create(batch_size, depth, height, width, data_type<T>, (cudnnTensorFormat_t)format);
    Data::Create(batch_size * depth * height * width * sizeof(T), data);
    this->batch_size = batch_size;
    this->depth = depth;
    this->height = height;
    this->width = width;
  }
  void Destroy() {
    Data::Destroy();
    cudnn::TensorDescriptor::Destroy();
  }
};

template <typename T>
class Filter: public cudnn::FilterDescriptor, public Data {
public:
  int output_depth, input_depth, height, width;
  void Create(int output_depth, int input_depth, int height, int width, const void* data = NULL, Format format = NCHW) {
    cudnn::FilterDescriptor::Create(output_depth, input_depth, height, width, data_type<T>, (cudnnTensorFormat_t)format);
    Data::Create(output_depth * input_depth * height * width * sizeof(T), data);
    this->output_depth = output_depth;
    this->input_depth = input_depth;
    this->height = height;
    this->width = width;
  }
  void Destroy() {
    Data::Destroy();
    cudnn::FilterDescriptor::Destroy();
  }
};

template <typename T>
class CrossCorrelation: public cudnn::ConvolutionDescriptor {
  cudnn::Handle handle;
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  size_t workspace_size = 0;
  void* workspace_data_device = NULL;
public:
  void Create() {
    cudnn::ConvolutionDescriptor::Create(data_type<T>);
    handle.Create();
    convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }
  template <typename T2>
  void Configure(const Tensor<T2>& input, const Filter<T2>& filter, Tensor<T2>& output) {
    convolution_algorithm = FindAlgorithm(handle, input, filter, output);

    workspace_size = WorkspaceSize(handle, input, filter, output, convolution_algorithm);
    cuda_assert(cudaMalloc(&workspace_data_device, workspace_size));
  }
  template <typename T2>
  void* Run(const Tensor<T2>& input, const Filter<T2>& filter, Tensor<T2>& output) {
    Forward(
      handle,
      input, input.data,
      filter, filter.data,
      *this, convolution_algorithm, workspace_data_device, workspace_size,
      output, output.data
    );
    cudaDeviceSynchronize();
    return output.data;
  }
  void Destroy() {
    if (workspace_data_device) cuda_assert(cudaFree(workspace_data_device));
    handle.Destroy();
    cudnn::ConvolutionDescriptor::Destroy();
  }
};

