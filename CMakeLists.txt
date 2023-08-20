cmake_minimum_required(VERSION 3.10)
project(cuDNN)
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_subdirectory(cudart)
add_subdirectory(cuDNN)
add_subdirectory(CrossCorrelation)
add_subdirectory(MeasureSpeed)
add_subdirectory(Example)
