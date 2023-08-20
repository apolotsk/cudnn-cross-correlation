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

#include <string.h> // For `malloc()`.
#include <CrossCorrelation.hpp>
int main() {
  cv::Mat image = load_image("input.png");
  Tensor<float> input;
  input.Create(1, image.channels(), image.rows, image.cols, image.ptr(), NHWC);
  Filter<float> filter;
  filter.Create(1, input.depth, 3, 3, NULL);
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
  filter.CopyTo(filter_data());
  Tensor<float> output;
  output.Create(input.batch_size, filter.output_depth, input.height-filter.height+1, input.width-filter.width+1, NULL, NHWC);

  CrossCorrelation<float> cross_correlation;
  cross_correlation.Create();
  cross_correlation.Configure(input, filter, output);
  cross_correlation.Run(input, filter, output);

  void* output_data = malloc(output.size);
  output.CopyFrom(output_data);
  save_image(output_data, output.height, output.width, "output.png");
  free(output_data);

  cross_correlation.Destroy();
  output.Destroy();
  filter.Destroy();
  input.Destroy();
}
