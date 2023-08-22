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

#include <cstdio> // For printf.
#include <cstring> // For malloc, free.
#include <CrossCorrelation.hpp>
int main() {
  Tensor<float> input;
  {
    cv::Mat image = load_image("input.png");
    printf("Load an image from input.png.\n");
    input.Create(1, image.channels(), image.rows, image.cols, image.ptr(), Format::NHWC);
    printf("Create the input tensor with shape is [%d, %d, %d, %d] and image data.\n", input.BatchSize(), input.Height(), input.Width(), input.Depth());
  }

  Filter<float> filter;
  {
    float data[1][3][3][3] = {{
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
    printf("Initialize edge detection kernel.\n");
    filter.Create(1, input.Depth(), 3, 3, data);
    printf("Create the filter with shape [%d, %d, %d, %d] and kernel data.\n", filter.OutputDepth(), filter.InputDepth(), filter.Height(), filter.Width());
  }

  Tensor<float> output;
  {
    output.Create(input.BatchSize(), filter.OutputDepth(), input.Height()-filter.Height()+1, input.Width()-filter.Width()+1, nullptr, Format::NHWC);
    printf("Create the output tensor with shape [%d, %d, %d, %d].\n", output.BatchSize(), output.Height(), output.Width(), output.Depth());
  }

  CrossCorrelation<float> cross_correlation;
  {
    cross_correlation.Create();
    cross_correlation.Configure(input, filter, output);
    cross_correlation.Run(input, filter, output);
    printf("Cross-correlate the input tensor and the filter.\n");
  }

  {
    void* output_data = malloc(output.Size());
    output.CopyTo(output_data);
    save_image(output_data, output.Height(), output.Width(), "output.png");
    printf("Save the output tensor as an image to output.png.\n");
    free(output_data);
  }

  cross_correlation.Destroy();
  output.Destroy();
  filter.Destroy();
  input.Destroy();
}
