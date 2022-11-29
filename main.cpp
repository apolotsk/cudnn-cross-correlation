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

#include <opencv2/opencv.hpp>
cv::Mat load_image(const char* filepath) {
  cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  return image;
}

#include <opencv2/opencv.hpp>
void save_image(const void* data, int height, int width, const char* filepath) {
  cv::Mat image(height, width, CV_32FC1, (void*)data);
  cv::threshold(image, image, 0, 0, cv::THRESH_TOZERO);
  cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
  image.convertTo(image, CV_8UC1);
  cv::imwrite(filepath, image);
}

#include <tuple>
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

typedef __fp16 half;
template <typename T> cudnnDataType_t data_type;
template <> cudnnDataType_t data_type<float> = CUDNN_DATA_FLOAT;
template <> cudnnDataType_t data_type<half> = CUDNN_DATA_HALF;

template <typename T>
class Tensor: public TensorDescriptor {
public:
  int batch_size, depth, height, width;
  void* data;

  void Create(int batch_size, int depth, int height, int width, const void* data = NULL, cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC) {
    TensorDescriptor::Create(batch_size, depth, height, width, data_type<T>, format);
    this->batch_size = batch_size;
    this->depth = depth;
    this->height = height;
    this->width = width;
    cuda_assert(cudaMalloc(&this->data, Size()));
    if (data) SetData(data);
  }
  int Size() const { return batch_size * depth * height * width * sizeof(T); }
  void SetData(const void* data) {
    cuda_assert(cudaMemcpy(this->data, data, Size(), cudaMemcpyHostToDevice));
  }
  void* Data(void* data) const {
    cuda_assert(cudaMemcpy(data, this->data, Size(), cudaMemcpyDeviceToHost));
    return data;
  }
  void Destroy() {
    cuda_assert(cudaFree(data));
    TensorDescriptor::Destroy();
  }
};

template <typename T>
class Filter: public FilterDescriptor {
public:
  int output_depth, input_depth, height, width;
  void* data;

  void Create(int output_depth, int input_depth, int height, int width, const void* data = NULL, cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC) {
    FilterDescriptor::Create(output_depth, input_depth, height, width, data_type<T>, format);
    this->output_depth = output_depth;
    this->input_depth = input_depth;
    this->height = height;
    this->width = width;
    cuda_assert(cudaMalloc(&this->data, Size()));
    if (data) SetData(data);
  }
  int Size() const { return output_depth * input_depth * height * width * sizeof(T); }
  void SetData(const void* data) {
    cuda_assert(cudaMemcpy(this->data, data, Size(), cudaMemcpyHostToDevice));
  }
  void* Data(void* data) const {
    cuda_assert(cudaMemcpy(data, this->data, Size(), cudaMemcpyDeviceToHost));
    return data;
  }
  void Destroy() {
    cuda_assert(cudaFree(data));
    FilterDescriptor::Destroy();
  }
};

template <typename T>
class CrossCorrelation: public ConvolutionDescriptor {
  Handle handle;
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  size_t workspace_size = 0;
  void* workspace_data_device = NULL;
public:
  void Create() {
    ConvolutionDescriptor::Create(data_type<T>);
    handle.Create();
    convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }
  void Configure(const Tensor<T>& input, const Filter<T>& filter, Tensor<T>& output) {
    convolution_algorithm = FindAlgorithm(handle, input, filter, output);
    printf("convolution_algorithm = %d\n", convolution_algorithm);

    workspace_size = WorkspaceSize(handle, input, filter, output, convolution_algorithm);
    printf("workspace_size = %lu\n", workspace_size);
    cuda_assert(cudaMalloc(&workspace_data_device, workspace_size));
  }
  void* Run(const Tensor<T>& input, const Filter<T>& filter, Tensor<T>& output) {
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
    ConvolutionDescriptor::Destroy();
  }
};

int main() {
  cv::Mat image = load_image("input.png");
  Tensor<float> input;
  input.Create(1, image.channels(), image.rows, image.cols, image.ptr(), CUDNN_TENSOR_NHWC);
  Filter<float> filter;
  filter.Create(1, input.depth, 3, 3, NULL, CUDNN_TENSOR_NCHW);
  auto filter_data = []()->const void* {
    static float data[1][3][3][3] = {{
      {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1},
      },
      {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1},
      },
      {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1},
      }
    }};
    return data;
  };
  filter.SetData(filter_data());
  Tensor<float> output;
  output.Create(input.batch_size, filter.output_depth, input.height-filter.height+1, input.width-filter.width+1, NULL, CUDNN_TENSOR_NHWC);

  CrossCorrelation<float> cross_correlation;
  cross_correlation.Create();
  cross_correlation.Configure(input, filter, output);
  cross_correlation.Run(input, filter, output);

  void* output_data = malloc(output.Size());
  output.Data(output_data);
  save_image(output_data, output.height, output.width, "output.png");
  free(output_data);

  cross_correlation.Destroy();
  output.Destroy();
  filter.Destroy();
  input.Destroy();
}
