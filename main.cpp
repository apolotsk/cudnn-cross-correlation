// https://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
#include <stdio.h> // For `printf()`.
#include <assert.h>
#include <stdlib.h> // For `exit()`.

#include <cuda_runtime.h>
void cuda_assert(cudaError_t error) {
  if (error==cudaSuccess) return;
  printf("Error: %s.", cudaGetErrorString(error));
  exit(1);
}

#include <cudnn.h>
void cudnn_assert(cudnnStatus_t status) {
  if (status==CUDNN_STATUS_SUCCESS) return;
  printf("Error: %s.", cudnnGetErrorString(status));
  exit(1);
}

#include <opencv2/opencv.hpp>
cv::Mat load_image(const char* filepath) {
  cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  return image;
}

#include <opencv2/opencv.hpp>
void save_image(const float* data, int height, int width, const char* filepath) {
  cv::Mat image(height, width, CV_32FC1, (float*)data);
  cv::threshold(image, image, 0, 0, cv::THRESH_TOZERO);
  cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
  image.convertTo(image, CV_8UC1);
  cv::imwrite(filepath, image);
}

int main() {
  cv::Mat image = load_image("input.png");
  void* input_data = image.ptr();
  int batch_size = 1, input_channels = image.channels(), input_height = image.rows, input_width = image.cols;

  cudnnHandle_t handle;
  cudnnCreate(&handle);

  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnn_assert(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  cudnn_assert(cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  cudnnTensorDescriptor_t input_descriptor;
  cudnn_assert(cudnnCreateTensorDescriptor(&input_descriptor));
  cudnn_assert(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, input_channels, input_height, input_width));

  cudnnFilterDescriptor_t filter_descriptor;
  cudnn_assert(cudnnCreateFilterDescriptor(&filter_descriptor));
  const int filter_output_count = 1, filter_input_count = input_channels, filter_height = 3, filter_width = 3;
  cudnn_assert(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_output_count, filter_input_count, filter_height, filter_width));

  int output_batch_size, output_channels, output_height, output_width;
  cudnn_assert(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &output_batch_size, &output_channels, &output_height, &output_width));
  assert(output_batch_size==batch_size);
  assert(output_channels==filter_output_count);
  cudnnTensorDescriptor_t output_descriptor;
  cudnn_assert(cudnnCreateTensorDescriptor(&output_descriptor));
  cudnn_assert(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, output_channels, output_height, output_width));

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  {
    cudnnConvolutionFwdAlgoPerf_t performance_result;
    int count;
    cudnn_assert(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 1, &count, &performance_result));
    convolution_algorithm = performance_result.algo;
  }

  size_t workspace_size = 0;
  cudnn_assert(cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_size));
  void* workspace_data_device = NULL;
  cuda_assert(cudaMalloc(&workspace_data_device, workspace_size));

  float* input_data_device = NULL;
  {
    int input_data_size = batch_size * input_channels * input_height * input_width * sizeof(float);
    cuda_assert(cudaMalloc(&input_data_device, input_data_size));
    cuda_assert(cudaMemcpy(input_data_device, input_data, input_data_size, cudaMemcpyHostToDevice));
  }

  float* output_data_device = NULL;
  {
    int output_data_size = batch_size * output_channels * output_height * output_width * sizeof(float);
    cuda_assert(cudaMalloc(&output_data_device, output_data_size));
    cuda_assert(cudaMemset(output_data_device, 0, output_data_size));
  }

  float* filter_data_device = NULL;
  {
    float filter_data[filter_output_count][filter_input_count][filter_height][filter_width];
    const float filter_template[filter_height][filter_width] = {
      {1, 1, 1},
      {1, -8, 1},
      {1, 1, 1}
    };

    for (int o = 0; o<filter_output_count; ++o) {
      for (int i = 0; i<filter_input_count; ++i) {
        memcpy(filter_data[o][i], filter_template, sizeof filter_template);
      }
    }

    cuda_assert(cudaMalloc(&filter_data_device, sizeof filter_data));
    cuda_assert(cudaMemcpy(filter_data_device, filter_data, sizeof filter_data, cudaMemcpyHostToDevice));
  }

  const float alpha = 1.0f, beta = 0.0f;
  cudnn_assert(cudnnConvolutionForward(
    handle, &alpha,
    input_descriptor, input_data_device,
    filter_descriptor, filter_data_device,
    convolution_descriptor, convolution_algorithm, workspace_data_device, workspace_size,
    &beta,
    output_descriptor, output_data_device
  ));

  {
    int output_data_size = batch_size * output_channels * output_height * output_width * sizeof(float);
    float* output_data = new float[output_data_size];
    cuda_assert(cudaMemcpy(output_data, output_data_device, output_data_size, cudaMemcpyDeviceToHost));
    save_image(output_data, output_height, output_width, "output.png");
    delete[] output_data;
  }

  cuda_assert(cudaFree(filter_data_device));
  cuda_assert(cudaFree(output_data_device));
  cuda_assert(cudaFree(input_data_device));
  cuda_assert(cudaFree(workspace_data_device));

  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(filter_descriptor);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(handle);
}
