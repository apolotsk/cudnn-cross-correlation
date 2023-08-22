#include <stdio.h> // For printf.
#include <string.h> // For malloc, free.
#include <CrossCorrelation.hpp>
#include "Stopwatch.hpp"
#include "Rand.hpp"

int main() {
  typedef float type;
  Tensor<type> input;
  input.Create(1, 1, 128, 128, NULL, NCHW);
  void* input_data = malloc(input.size);
  rand<type>(input_data, input.size/sizeof(type));
  input.CopyTo(input_data);
  free(input_data);
  printf("Create input tensor with shape [%d, %d, %d, %d] and random data.\n", input.batch_size, input.depth, input.height, input.width);

  Filter<type> filter;
  filter.Create(512, input.depth, 16, 16, NULL, NCHW);
  void* filter_data = malloc(filter.size);
  rand<type>(filter_data, filter.size/sizeof(type));
  filter.CopyTo(filter_data);
  free(filter_data);
  printf("Create filter with shape [%d, %d, %d, %d] and random data.\n", filter.output_depth, filter.input_depth, filter.height, filter.width);

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
  printf("Cross-correlation takes %.1f ms.\n", stopwatch.Time()/count*1e3);

  cross_correlation.Destroy();
  output.Destroy();
  filter.Destroy();
  input.Destroy();
}
