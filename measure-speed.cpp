// https://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
#include <stdio.h> // For `printf()`.
#include <assert.h>
#include <string.h> // For `memcpy()`.
#include <stdlib.h> // For `exit()`.

#include <cuda_runtime.h>
void _cuda_assert(cudaError_t error, int line) {
  if (error==cudaSuccess) return;
  printf("Error at line %d: %s.\n", line, cudaGetErrorString(error));
  exit(1);
}
#define cuda_assert(status) _cuda_assert(status, __LINE__);

#include <cudnn.h>
void _cudnn_assert(cudnnStatus_t status, int line) {
  if (status==CUDNN_STATUS_SUCCESS) return;
  printf("Error at line %d: %s.\n", line, cudnnGetErrorString(status));
  exit(1);
}
#define cudnn_assert(status) _cudnn_assert(status, __LINE__);

#include <Stopwatch.hpp>
#include <Rand.hpp>
int main() {
  cudnnDataType_t type = CUDNN_DATA_HALF;
  typedef half Type;

  cudnnConvolutionDescriptor_t convolution_descriptor;
  {
    cudnn_assert(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    cudnn_assert(cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, type));
  }

  const int batch_size = 1, input_channels = 1, input_height = 128, input_width = 128;
  cudnnTensorDescriptor_t input_descriptor;
  {
    cudnn_assert(cudnnCreateTensorDescriptor(&input_descriptor));
    cudnn_assert(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, type, batch_size, input_channels, input_height, input_width));
  }

  const int filter_output_count = 512, filter_input_count = input_channels, filter_height = 16, filter_width = 16;
  cudnnFilterDescriptor_t filter_descriptor;
  {
    cudnn_assert(cudnnCreateFilterDescriptor(&filter_descriptor));
    cudnn_assert(cudnnSetFilter4dDescriptor(filter_descriptor, type, CUDNN_TENSOR_NCHW, filter_output_count, filter_input_count, filter_height, filter_width));
  }

  int output_batch_size, output_channels, output_height, output_width;
  {
    cudnn_assert(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &output_batch_size, &output_channels, &output_height, &output_width));
    assert(output_batch_size==batch_size);
    assert(output_channels==filter_output_count);
  }
  cudnnTensorDescriptor_t output_descriptor;
  {
    cudnn_assert(cudnnCreateTensorDescriptor(&output_descriptor));
    cudnn_assert(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, type, batch_size, output_channels, output_height, output_width));
  }

  cudnnHandle_t handle;
  {
    cudnnCreate(&handle);
  }

  cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  if (1) {
    cudnnConvolutionFwdAlgoPerf_t performance_result;
    int count;
    cudnn_assert(cudnnFindConvolutionForwardAlgorithm(handle, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 1, &count, &performance_result));
    convolution_algorithm = performance_result.algo;
  }
  printf("convolution_algorithm = %d\n", convolution_algorithm);

  size_t workspace_size = 0;
  {
    cudnn_assert(cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_size));
    printf("workspace_size = %lu\n", workspace_size);
  }
  void* workspace_data_device = NULL;
  {
    cuda_assert(cudaMalloc(&workspace_data_device, workspace_size));
  }

  void* input_data = NULL;
  void* input_data_device = NULL;
  {
    int input_data_size = batch_size * input_channels * input_height * input_width * sizeof(Type);
    input_data = malloc(input_data_size);
    rand<Type>(input_data, input_data_size/sizeof(Type));
    cuda_assert(cudaMalloc(&input_data_device, input_data_size));
    cuda_assert(cudaMemcpy(input_data_device, input_data, input_data_size, cudaMemcpyHostToDevice));
  }

  void *filter_data = NULL;
  void* filter_data_device = NULL;
  {
    int count = filter_output_count * filter_input_count * filter_height * filter_width;
    int size = count * sizeof(Type);
    filter_data = malloc(size);
    rand<Type>(filter_data, count);
    cuda_assert(cudaMalloc(&filter_data_device, size));
    cuda_assert(cudaMemcpy(filter_data_device, filter_data, size, cudaMemcpyHostToDevice));
  }

  void *output_data = NULL;
  void* output_data_device = NULL;
  {
    int output_data_size = batch_size * output_channels * output_height * output_width * sizeof(Type);
    output_data = malloc(output_data_size);
    cuda_assert(cudaMalloc(&output_data_device, output_data_size));
    cuda_assert(cudaMemset(output_data_device, 0, output_data_size));
  }


  auto run = [&]() {
    const float alpha = 1.0f, beta = 0.0f;
    cudnn_assert(cudnnConvolutionForward(
      handle, &alpha,
      input_descriptor, input_data_device,
      filter_descriptor, filter_data_device,
      convolution_descriptor, convolution_algorithm, workspace_data_device, workspace_size,
      &beta,
      output_descriptor, output_data_device
    ));
    cudaDeviceSynchronize();
  };
  run();


  Stopwatch stopwatch;
  const int count = 10;
  for (int i = 0; i<count; ++i) {
    run();
  }
  printf("time = %.2f ms\n", stopwatch.Time()/count*1e3);

  {
    int output_data_size = batch_size * output_channels * output_height * output_width * sizeof(Type);
    cuda_assert(cudaMemcpy(output_data, output_data_device, output_data_size, cudaMemcpyDeviceToHost));
  }

  cuda_assert(cudaFree(output_data_device));
  free(output_data);
  cuda_assert(cudaFree(filter_data_device));
  free(filter_data);
  cuda_assert(cudaFree(input_data_device));
  free(input_data);
  cuda_assert(cudaFree(workspace_data_device));

  cudnnDestroy(handle);

  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(filter_descriptor);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}
