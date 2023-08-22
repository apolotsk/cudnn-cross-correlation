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

#include <stdio.h> // For printf.
#include <string.h> // For malloc, free.
#include <CrossCorrelation.hpp>
int main() {
  cv::Mat image = load_image("input.png");
  printf("Load an image from input.png.\n");

  Tensor<float> input;
  input.Create(1, image.channels(), image.rows, image.cols, image.ptr(), NHWC);
  printf("Create the input tensor.\n");
  printf("Copy the image into the input tensor.\n");
  printf("The input tensor shape is [%d, %d, %d, %d].\n", input.batch_size, input.height, input.width, input.depth);

  Filter<float> filter;
  filter.Create(1, input.depth, 3, 3, NULL);
  printf("Create the filter.\n");
  printf("The filter shape is [%d, %d, %d, %d].\n", filter.output_depth, filter.input_depth, filter.height, filter.width);
  auto filter_data = []()->const void* {
    static float data[1][3][3][3] = {{
      {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1},
      },
      {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1},
      },
      {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1},
      }
    }};
    return data;
  };
  printf("Create edge detection kernel data.\n");
  filter.CopyTo(filter_data());
  printf("Copy the kernel data into the filter.\n");

  Tensor<float> output;
  output.Create(input.batch_size, filter.output_depth, input.height-filter.height+1, input.width-filter.width+1, NULL, NHWC);
  printf("Create the output tensor.\n");
  printf("The output tensor shape [%d, %d, %d, %d].\n", output.batch_size, output.height, output.width, output.depth);

  CrossCorrelation<float> cross_correlation;
  cross_correlation.Create();
  cross_correlation.Configure(input, filter, output);
  cross_correlation.Run(input, filter, output);
  printf("Cross-correlate the input tensor and the filter.\n");

  void* output_data = malloc(output.Size());
  output.CopyFrom(output_data);
  save_image(output_data, output.height, output.width, "output.png");
  printf("Save the output tensor as an image to output.png.\n");
  free(output_data);

  cross_correlation.Destroy();
  output.Destroy();
  filter.Destroy();
  input.Destroy();
}
