#include <string.h> // For `malloc()`.
#include <Stopwatch.hpp>
#include <Rand.hpp>
#include "Convolution.hpp"
int main() {
  typedef float type;
  Tensor<type> input;
  input.Create(1, 1, 128, 128, NULL, NCHW);
  void* input_data = malloc(input.size);
  rand<type>(input_data, input.size/sizeof(type));
  input.CopyTo(input_data);
  free(input_data);

  Filter<type> filter;
  filter.Create(512, input.depth, 16, 16, NULL, NCHW);
  void* filter_data = malloc(filter.size);
  rand<type>(filter_data, filter.size/sizeof(type));
  filter.CopyTo(filter_data);
  free(filter_data);

  Tensor<type> output;
  output.Create(input.batch_size, filter.output_depth, input.height-filter.height+1, input.width-filter.width+1, NULL, NCHW);

  CrossCorrelation<float> cross_correlation;
  cross_correlation.Create();
  cross_correlation.Configure(input, filter, output);
  cross_correlation.Run(input, filter, output);

  Stopwatch stopwatch;
  const int count = 10;
  for (int i = 0; i<count; ++i) {
    cross_correlation.Run(input, filter, output);
  }
  printf("time = %.1f ms\n", stopwatch.Time()/count*1e3); // Jetson Nano: 27.2 ms.

  cross_correlation.Destroy();
  output.Destroy();
  filter.Destroy();
  input.Destroy();
}
