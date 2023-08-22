#pragma once
#include <cudart.hpp> // For cuda_assert.
#include <cuDNN.hpp> // For cuDNN::*.

typedef cuDNN::Format Format;

/** \brief A tensor (the description and the data). */
template <typename T>
class Tensor: public cuDNN::TensorDescriptor, public cudart::DeviceData {
public:
  void Create(int batch_size, int depth, int height, int width, const void* data = nullptr, Format format = Format::NCHW) {
    cuDNN::TensorDescriptor::Create<T>(batch_size, depth, height, width, format);
    DeviceData::Create(batch_size * depth * height * width * sizeof(T), data);
  }
  void Destroy() {
    DeviceData::Destroy();
    cuDNN::TensorDescriptor::Destroy();
  }
};

/** \brief A filter (the description and the data). */
template <typename T>
class Filter: public cuDNN::FilterDescriptor, public cudart::DeviceData {
public:
  void Create(int output_depth, int input_depth, int height, int width, const void* data = nullptr, Format format = Format::NCHW) {
    cuDNN::FilterDescriptor::Create<T>(output_depth, input_depth, height, width, format);
    DeviceData::Create(output_depth * input_depth * height * width * sizeof(T), data);
  }
  void Destroy() {
    DeviceData::Destroy();
    cuDNN::FilterDescriptor::Destroy();
  }
};

/** \brief A Cross-collelation. */
template <typename T>
class CrossCorrelation: public cuDNN::ConvolutionDescriptor {
  cuDNN::Handle handle;
  cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  cudart::DeviceData workspace;
public:
  void Create() {
    cuDNN::ConvolutionDescriptor::Create<T>(CUDNN_CROSS_CORRELATION);
    handle.Create();
  }
  /** \brief Configure the cross-correlation given the input tensor, filter and output tensor. */
  void Configure(const Tensor<T>& input, const Filter<T>& filter, const Tensor<T>& output) {
    convolution_algorithm = FindAlgorithm(handle, input, filter, output);
    workspace.Create(WorkspaceSize(handle, input, filter, output, convolution_algorithm));
  }
  template <typename T2>
  void* Run(const Tensor<T2>& input, const Filter<T2>& filter, Tensor<T2>& output) {
    Forward(
      handle,
      input, input.Data(),
      filter, filter.Data(),
      *this, convolution_algorithm, workspace.Data(), workspace.Size(),
      output, output.Data()
    );
    cudaDeviceSynchronize();
    return output.Data();
  }
  void Destroy() {
    workspace.Destroy();
    handle.Destroy();
    cuDNN::ConvolutionDescriptor::Destroy();
  }
};
