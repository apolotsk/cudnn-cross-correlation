cmake_minimum_required(VERSION 3.10)
project("cuDNN CrossCorrelation")
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_subdirectory(MeasureSpeed)
add_subdirectory(Example)
