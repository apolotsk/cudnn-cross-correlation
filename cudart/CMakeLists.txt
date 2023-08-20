find_package(PkgConfig REQUIRED)
pkg_check_modules(cudart REQUIRED IMPORTED_TARGET cudart-10.2)

add_library(cudart INTERFACE)# cudart.hpp)
target_include_directories(cudart INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(cudart INTERFACE PkgConfig::cudart)
