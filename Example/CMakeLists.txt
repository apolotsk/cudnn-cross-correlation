find_package(PkgConfig REQUIRED)
pkg_check_modules(opencv REQUIRED IMPORTED_TARGET opencv4)

add_executable(Example Example.cpp)
target_link_libraries(Example PUBLIC Convolution PkgConfig::opencv)
configure_file(input.png ${CMAKE_CURRENT_BINARY_DIR}/input.png COPYONLY)
