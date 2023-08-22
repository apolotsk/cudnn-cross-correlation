find_package(PkgConfig REQUIRED)
pkg_check_modules(cudart REQUIRED IMPORTED_TARGET cudart-10.2)

add_executable(MeasureSpeed MeasureSpeed.cpp)
target_include_directories(MeasureSpeed PUBLIC ../cuda ../cudnn .)
target_link_libraries(MeasureSpeed PUBLIC PkgConfig::cudart cudnn)
