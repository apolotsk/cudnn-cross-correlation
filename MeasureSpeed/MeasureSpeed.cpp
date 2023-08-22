#include <cstdio> // For printf.
#include <cstring> // For malloc, free.
#include <CrossCorrelation.hpp>

#include <cstdlib> // For rand.
template <typename T> T rand();
template<> inline double rand<double>() { return (double)rand()/RAND_MAX; }
template<> inline float rand<float>() { return (float)rand<double>(); }
template<> inline half rand<half>() { return (half)rand<double>(); }
template <typename T>
void rand(void* data, int count) {
  for (int i = 0; i<count; ++i) ((T*)data)[i] = rand<T>();
}

#include <chrono> // For `std::chrono`.
double timestamp() {
  std::chrono::steady_clock::time_point time_point = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(time_point.time_since_epoch()).count();
}

int main() {
  typedef float type;
  Tensor<type> input;
  {
    input.Create(1, 1, 128, 128, nullptr, NCHW);
    void* input_data = malloc(input.Size());
    rand<type>(input_data, input.Size()/sizeof(type));
    input.CopyFrom(input_data);
    free(input_data);
    printf("Create the input tensor with shape [%d, %d, %d, %d] and random data.\n", input.batch_size, input.depth, input.height, input.width);
  }

  Filter<type> filter;
  {
    filter.Create(512, input.depth, 16, 16, nullptr, NCHW);
    void* filter_data = malloc(filter.Size());
    rand<type>(filter_data, filter.Size()/sizeof(type));
    filter.CopyFrom(filter_data);
    free(filter_data);
    printf("Create the filter with shape [%d, %d, %d, %d] and random data.\n", filter.output_depth, filter.input_depth, filter.height, filter.width);
  }

  Tensor<type> output;
  {
    output.Create(input.batch_size, filter.output_depth, input.height-filter.height+1, input.width-filter.width+1, nullptr, NCHW);
    printf("Create the output tensor with shape [%d, %d, %d, %d].\n", output.batch_size, output.depth, output.height, output.width);
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
