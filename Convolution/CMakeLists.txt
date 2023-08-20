add_library(Convolution STATIC Convolution.hpp)
set_target_properties(Convolution PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(Convolution INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(Convolution INTERFACE cudart cuDNN)
