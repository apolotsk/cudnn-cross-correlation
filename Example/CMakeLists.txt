find_package(PkgConfig REQUIRED)
pkg_check_modules(cudart REQUIRED IMPORTED_TARGET cudart-10.2)
pkg_check_modules(opencv REQUIRED IMPORTED_TARGET opencv4)

add_executable(Example Example.cpp)
target_link_libraries(Example PUBLIC PkgConfig::opencv PkgConfig::cudart cudnn)
configure_file(input.png ${CMAKE_CURRENT_BINARY_DIR}/input.png COPYONLY)
target_include_directories(Example PUBLIC ../cuda ../cudnn)
