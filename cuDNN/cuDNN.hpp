#pragma once
#include <stdio.h> // For `printf()`.
#include <stdexcept> // For std::runtime_error().
#include <stdlib.h> // For `exit()`.
#include <tuple>
#include <cudnn.h>

namespace cuDNN {

void _cudnn_assert(cudnnStatus_t status, const char* call_file, unsigned int call_line, const char* expression) {
  if (status==CUDNN_STATUS_SUCCESS) return;
  printf("Assertion in %s:%d %s. %s\n", call_file, call_line, expression, cudnnGetErrorString(status));
  throw std::runtime_error(cudnnGetErrorString(status));
}
#define cudnn_assert(expr) _cudnn_assert(expr, __FILE__, __LINE__, #expr);

class ConvolutionDescriptor {
  cudnnConvolutionDescriptor_t convolution_descriptor;
public:
  void Create(cudnnDataType_t type = CUDNN_DATA_FLOAT) {
    cudnn_assert(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    cudnn_assert(cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, type));
  }
  std::tuple<int,int,int,int> OutputDim(const cudnnTensorDescriptor_t& input_descriptor, const cudnnFilterDescriptor_t& filter_descriptor) {
    int batch_size, channels, height, width;
    cudnn_assert(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &batch_size, &channels, &height, &width));
    return {batch_size, channels, height, width};
  }
  cudnnConvolutionFwdAlgo_t FindAlgorithm(const cudnnHandle_t& handle, const cudnnTensorDescriptor_t& input_descriptor, const cudnnFilterDescriptor_t& filter_descriptor, const cudnnTensorDescriptor_t& output_descriptor) {
    cudnnConvolutionFwdAlgoPerf_t performance_result;
    int count;
    cudnn_assert(cudnnFindConvolutionForwardAlgorithm(handle, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 1, &count, &performance_result));
    return performance_result.algo;
  }
  int WorkspaceSize(cudnnHandle_t handle, cudnnTensorDescriptor_t input_descriptor, cudnnFilterDescriptor_t filter_descriptor, cudnnTensorDescriptor_t output_descriptor, cudnnConvolutionFwdAlgo_t convolution_algorithm) {
    size_t workspace_size;
    cudnn_assert(cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_size));
    return workspace_size;
  }
  void Forward(cudnnHandle_t handle, const cudnnTensorDescriptor_t input_descriptor, const void *input_data_device, const cudnnFilterDescriptor_t filter_descriptor, const void *filter_data_device, const cudnnConvolutionDescriptor_t convolution_descriptor, cudnnConvolutionFwdAlgo_t convolution_algorithm, void *workspace_data_device, size_t workspace_size, const cudnnTensorDescriptor_t output_descriptor, void *output_data_device) {
    const float alpha = 1.0f, beta = 0.0f;
    cudnn_assert(cudnnConvolutionForward(
      handle, &alpha,
      input_descriptor, input_data_device,
      filter_descriptor, filter_data_device,
      convolution_descriptor, convolution_algorithm, workspace_data_device, workspace_size,
      &beta,
      output_descriptor, output_data_device
    ));
  }
  operator cudnnConvolutionDescriptor_t() const { return convolution_descriptor; }
  void Destroy() {
    cudnn_assert(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
  }
};

class TensorDescriptor {
  cudnnTensorDescriptor_t tensor_descriptor;
public:
  void Create(int batch_size, int channels, int height, int width, cudnnDataType_t type = CUDNN_DATA_FLOAT, cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC) {
    cudnn_assert(cudnnCreateTensorDescriptor(&tensor_descriptor));
    cudnn_assert(cudnnSetTensor4dDescriptor(tensor_descriptor, format, type, batch_size, channels, height, width));
  }
  operator cudnnTensorDescriptor_t() const { return tensor_descriptor; }
  void Destroy() {
    cudnn_assert(cudnnDestroyTensorDescriptor(tensor_descriptor));
  }
};

class FilterDescriptor {
  cudnnFilterDescriptor_t filter_descriptor;
public:
  void Create(int output_count, int input_count, int height, int width, cudnnDataType_t type = CUDNN_DATA_FLOAT, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
    cudnn_assert(cudnnCreateFilterDescriptor(&filter_descriptor));
    cudnn_assert(cudnnSetFilter4dDescriptor(filter_descriptor, type, format, output_count, input_count, height, width));
  }
  operator cudnnFilterDescriptor_t() const { return filter_descriptor; }
  void Destroy() {
    cudnn_assert(cudnnDestroyFilterDescriptor(filter_descriptor));
  }
};

class Handle {
  cudnnHandle_t handle;
public:
  void Create() {
    cudnnCreate(&handle);
  }
  operator cudnnHandle_t() const { return handle; }
  void Destroy() {
    cudnn_assert(cudnnDestroy(handle));
  }
};

}
