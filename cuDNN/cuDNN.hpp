#pragma once
#include <cstdio> // For printf.
#include <stdexcept> // For std::runtime_error().
#include <tuple>
#include <cudnn.h>

typedef __fp16 half;

/** \brief <a href="https://docs.nvidia.com/deeplearning/cudnn/api/index.html">cuDNN API</a> for C++. */
namespace cuDNN {

/**
 * \brief Assert a cuDNN function call.
 * 
 * If there is an error, then throw a runtime error.
 *
 * \param expression A cuDNN function call. For example, `cudnnCreate()`.
 */
#define cudnn_assert(expression) _cudnn_assert(expression, __FILE__, __LINE__, #expression);
/** \brief Is intended to be called by the macro `cudnn_assert` only. */
void _cudnn_assert(cudnnStatus_t status, const char* call_file, unsigned int call_line, const char* expression) {
  if (status==CUDNN_STATUS_SUCCESS) return;
  printf("Assertion in %s:%d %s. %s\n", call_file, call_line, expression, cudnnGetErrorString(status));
  throw std::runtime_error(cudnnGetErrorString(status));
}

template <typename T> cudnnDataType_t type;
template <> cudnnDataType_t type<half> = CUDNN_DATA_HALF;
template <> cudnnDataType_t type<float> = CUDNN_DATA_FLOAT;
template <> cudnnDataType_t type<double> = CUDNN_DATA_DOUBLE;

/** \brief Tensor format. */
enum Format {
  /** Layout is batch_size, depth, height, width. */
  NCHW = CUDNN_TENSOR_NCHW,
  /** Layout is batch_size, height, width, depth. */
  NHWC = CUDNN_TENSOR_NHWC,
};

/** \brief A tensor desciption (shape, type, layout, does not include data). */
class TensorDescriptor {
  /** \brief  pointer to the opaque tensor description. */
  cudnnTensorDescriptor_t tensor_descriptor;

  struct Parameters {
    cudnnDataType_t type;
    int batch_size, depth, height, width;
    int batch_size_stride, depth_stride, height_stride, width_stride;
  };
  /** \brief Queriy the parameters. */
  Parameters GetParameters() const {
    Parameters p;
    cudnn_assert(cudnnGetTensor4dDescriptor(tensor_descriptor,
      &p.type,
      &p.batch_size, &p.depth, &p.height, &p.width,
      &p.batch_size_stride, &p.depth_stride, &p.height_stride, &p.width_stride
    ));
    return p;
  }

public:
  /** \brief Create and initialize the tensor descriptor. */
  void Create(int batch_size, int depth, int height, int width, cudnnDataType_t type = CUDNN_DATA_FLOAT, cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC) {
    cudnn_assert(cudnnCreateTensorDescriptor(&tensor_descriptor));
    cudnn_assert(cudnnSetTensor4dDescriptor(tensor_descriptor, format, type, batch_size, depth, height, width));
  }
  /** \brief Create and initialize the tensor descriptor. */
  template <typename T>
  void Create(int batch_size, int depth, int height, int width, Format format = NHWC) {
    Create(batch_size, depth, height, width, type<T>, (cudnnTensorFormat_t)format);
  }
  operator cudnnTensorDescriptor_t() const { return tensor_descriptor; }

  cudnnDataType_t Type() const { return GetParameters().type; }
  int BatchSize() const { return GetParameters().batch_size; }
  int Depth() const { return GetParameters().depth; }
  int Height() const { return GetParameters().height; }
  int Width() const { return GetParameters().width; }

  /** \brief Destroy the tensor descriptor. */
  void Destroy() {
    cudnn_assert(cudnnDestroyTensorDescriptor(tensor_descriptor));
  }
};

/** \brief A filter desciption (shape, type, layout, does not include data). */
class FilterDescriptor {
  /** \brief  pointer to the opaque filter description. */
  cudnnFilterDescriptor_t filter_descriptor;

  struct Parameters {
    cudnnDataType_t type;
    cudnnTensorFormat_t format;
    int output_depth, input_depth, height, width;
  };
  /** \brief Create and initialize the filter descriptor. */
  Parameters GetParameters() const {
    Parameters p;
    cudnn_assert(cudnnGetFilter4dDescriptor(filter_descriptor,
      &p.type, &p.format,
      &p.output_depth, &p.input_depth, &p.height, &p.width
    ));
    return p;
  }

public:
  /** \brief Create and initialize the filter descriptor. */
  void Create(int output_depth, int input_depth, int height, int width, cudnnDataType_t type = CUDNN_DATA_FLOAT, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
    cudnn_assert(cudnnCreateFilterDescriptor(&filter_descriptor));
    cudnn_assert(cudnnSetFilter4dDescriptor(filter_descriptor, type, format, output_depth, input_depth, height, width));
  }
  /** \brief Create and initialize the filter descriptor. */
  template <typename T>
  void Create(int output_depth, int input_depth, int height, int width, Format format = NCHW) {
    Create(output_depth, input_depth, height, width, type<T>, (cudnnTensorFormat_t)format);
  }
  operator cudnnFilterDescriptor_t() const { return filter_descriptor; }

  cudnnDataType_t Type() const { return GetParameters().type; }
  cudnnTensorFormat_t Format_() const { return GetParameters().format; }
  int OutputDepth() const { return GetParameters().output_depth; }
  int InputDepth() const { return GetParameters().input_depth; }
  int Height() const { return GetParameters().height; }
  int Width() const { return GetParameters().width; }

  /** \brief Destroy the filter descriptor. */
  void Destroy() {
    cudnn_assert(cudnnDestroyFilterDescriptor(filter_descriptor));
  }
};

/** \brief A handle to the cuDNN library context. */
class Handle {
  cudnnHandle_t handle;
public:
  /** \brief Create the handle. */
  void Create() {
    cudnn_assert(cudnnCreate(&handle));
  }
  operator cudnnHandle_t() const { return handle; }
  /** \brief Destroy the handle. */
  void Destroy() {
    cudnn_assert(cudnnDestroy(handle));
  }
};

/** \brief A convolution desciption (padding, strides, dilation, type, does not include tensor description or data). */
class ConvolutionDescriptor {
  cudnnConvolutionDescriptor_t convolution_descriptor;
public:
  /** \brief Create and initialize the convolution descriptor. */
  void Create(cudnnDataType_t type = CUDNN_DATA_FLOAT, cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION) {
    cudnn_assert(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    cudnn_assert(cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, mode, type));
  }
  /** \brief Create and initialize the convolution descriptor. */
  template <typename T>
  void Create(cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION) {
    Create(type<T>, mode);
  }
  /** \brief Compute the output tensor shape given the input tensor and filter descriptions. */
  std::tuple<int,int,int,int> OutputDim(const TensorDescriptor& input_descriptor, const FilterDescriptor& filter_descriptor) {
    int batch_size, depth, height, width;
    cudnn_assert(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &batch_size, &depth, &height, &width));
    return {batch_size, depth, height, width};
  }
  /** \brief Find a convolution implementation with the smallest compute time. */
  cudnnConvolutionFwdAlgo_t FindAlgorithm(const Handle& handle, const TensorDescriptor& input_descriptor, const FilterDescriptor& filter_descriptor, const cudnnTensorDescriptor_t& output_descriptor) {
    cudnnConvolutionFwdAlgoPerf_t performance_result;
    int count;
    cudnn_assert(cudnnFindConvolutionForwardAlgorithm(handle, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 1, &count, &performance_result));
    return performance_result.algo;
  }
  /** \brief Get the size needed for the `Forward` call. */
  int WorkspaceSize(const Handle& handle, const TensorDescriptor& input_descriptor, const FilterDescriptor& filter_descriptor, const cudnnTensorDescriptor_t& output_descriptor, const cudnnConvolutionFwdAlgo_t& convolution_algorithm) {
    size_t workspace_size;
    cudnn_assert(cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_size));
    return workspace_size;
  }
  /** \brief Run the convoltion. */
  void Forward(const Handle& handle, const TensorDescriptor& input_descriptor, const void *input_data_device, const FilterDescriptor filter_descriptor, const void *filter_data_device, const cudnnConvolutionDescriptor_t convolution_descriptor, cudnnConvolutionFwdAlgo_t convolution_algorithm, void *workspace_data_device, size_t workspace_size, const cudnnTensorDescriptor_t& output_descriptor, void *output_data_device) {
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
  /** \brief Destroy the convolution descriptor. */
  void Destroy() {
    cudnn_assert(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
  }
};

}
