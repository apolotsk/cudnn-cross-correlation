#include <cstdlib> // For rand.
template <typename T> inline T rand() { return (T)rand()/RAND_MAX; }
template <typename T> void rand(void* data, int count) {
  for (int i = 0; i<count; ++i) ((T*)data)[i] = rand<T>();
}

#include <chrono> // For std::chrono.
double timestamp() {
  std::chrono::steady_clock::time_point time_point = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(time_point.time_since_epoch()).count();
}

#include <cstdio> // For printf.
#include <CrossCorrelation.hpp>
int main() {
  typedef float type;
  Tensor<type> input;
  {
    input.Create(1, 1, 128, 128, nullptr, Format::NCHW);
    void* input_data = new char[input.Size()];
    rand<type>(input_data, input.Size()/sizeof(type));
    input.CopyFrom(input_data);
    delete [] (char*)input_data;
    printf("Create the input tensor with shape [%d, %d, %d, %d] and random data.\n", input.BatchSize(), input.Depth(), input.Height(), input.Width());
  }

  Filter<type> filter;
  {
    filter.Create(512, input.Depth(), 16, 16, nullptr, Format::NCHW);
    void* filter_data = new char[filter.Size()];
    rand<type>(filter_data, filter.Size()/sizeof(type));
    filter.CopyFrom(filter_data);
    delete [] (char*)filter_data;
    printf("Create the filter with shape [%d, %d, %d, %d] and random data.\n", filter.OutputDepth(), filter.InputDepth(), filter.Height(), filter.Width());
  }

  Tensor<type> output;
  {
    output.Create(input.BatchSize(), filter.OutputDepth(), input.Height()-filter.Height()+1, input.Width()-filter.Width()+1, nullptr, Format::NCHW);
    printf("Create the output tensor with shape [%d, %d, %d, %d].\n", output.BatchSize(), output.Depth(), output.Height(), output.Width());
  }

  CrossCorrelation<float> cross_correlation;
  {
    cross_correlation.Create();
    cross_correlation.Configure(input, filter, output);
    cross_correlation.Run(input, filter, output);

    double timestamp0 = timestamp();
    const int count = 10;
    for (int i = 0; i<count; ++i) {
      cross_correlation.Run(input, filter, output);
    }
    double timestamp1 = timestamp();
    printf("Cross-correlation takes %.1f ms (on average over %d runs).\n", (timestamp1-timestamp0)/count*1e3, count);
  }

  cross_correlation.Destroy();
  output.Destroy();
  filter.Destroy();
  input.Destroy();
}
