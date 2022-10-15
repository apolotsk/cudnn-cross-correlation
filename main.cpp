// https://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
#include <cuda_runtime.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>

void cuda_assert(cudaError_t error) {
  if (error==cudaSuccess) return;
  printf("Error: %s.", cudaGetErrorString(error));
  exit(1);
}

void cudnn_assert(cudnnStatus_t status) {
  if (status==CUDNN_STATUS_SUCCESS) return;
  printf("Error: %s.", cudnnGetErrorString(status));
  exit(1);
}

cv::Mat load_image(const char* filepath) {
  cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  return image;
}

void save_image(const float* data, int height, int width, const char* filepath) {
  cv::Mat image(height, width, CV_32FC3, (float*)data);
  cv::threshold(image, image, 0, 0, cv::THRESH_TOZERO);
  cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
  image.convertTo(image, CV_8UC3);
  cv::imwrite(filepath, image);
}

int main() {
  cv::Mat image = load_image("tensorflow.png");

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t tensor_descriptor;
  cudnn_assert(cudnnCreateTensorDescriptor(&tensor_descriptor));
  cudnn_assert(cudnnSetTensor4dDescriptor(tensor_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, image.rows, image.cols));

  cudnnFilterDescriptor_t filter_descriptor;
  cudnn_assert(cudnnCreateFilterDescriptor(&filter_descriptor));
  cudnn_assert(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, 3, 3, 3));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnn_assert(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  cudnn_assert(cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  int batch_size{0}, channels{0}, height{0}, width{0};
  cudnn_assert(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, tensor_descriptor, filter_descriptor, &batch_size, &channels, &height, &width));

  cudnnTensorDescriptor_t output_descriptor;
  cudnn_assert(cudnnCreateTensorDescriptor(&output_descriptor));
  cudnn_assert(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, image.rows, image.cols));

  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t perfResults[1];
  cudnn_assert(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, tensor_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 1, &returnedAlgoCount, perfResults));

  cudnnConvolutionFwdAlgo_t convolution_algorithm = perfResults[0].algo;
  size_t workspace_bytes{0};
  cudnn_assert(cudnnGetConvolutionForwardWorkspaceSize(cudnn, tensor_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_bytes));

  void* d_workspace{nullptr};
  cuda_assert(cudaMalloc(&d_workspace, workspace_bytes));

  int image_bytes = batch_size * channels * height * width * sizeof(float);

  float* d_input{nullptr};
  cuda_assert(cudaMalloc(&d_input, image_bytes));
  cuda_assert(cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice));

  float* d_output{nullptr};
  cuda_assert(cudaMalloc(&d_output, image_bytes));
  cuda_assert(cudaMemset(d_output, 0, image_bytes));

  const float kernel_template[3][3] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
  };

  float h_kernel[3][3][3][3];
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  float* d_kernel{nullptr};
  cuda_assert(cudaMalloc(&d_kernel, sizeof(h_kernel)));
  cuda_assert(cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice));

  const float alpha = 1.0f, beta = 0.0f;

  cudnn_assert(cudnnConvolutionForward(cudnn, &alpha, tensor_descriptor, d_input, filter_descriptor, d_kernel, convolution_descriptor, convolution_algorithm, d_workspace, workspace_bytes, &beta, output_descriptor, d_output));

  float* h_output = new float[image_bytes];
  cuda_assert(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));

  save_image(h_output, height, width, "cudnn-out.png");

  delete[] h_output;
  cuda_assert(cudaFree(d_kernel));
  cuda_assert(cudaFree(d_input));
  cuda_assert(cudaFree(d_output));
  cuda_assert(cudaFree(d_workspace));

  cudnnDestroyTensorDescriptor(tensor_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(filter_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}
