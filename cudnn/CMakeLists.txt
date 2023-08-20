add_library(cuDNN INTERFACE)# cudnn.hpp)
target_include_directories(cuDNN INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(cuDNN INTERFACE cudnn)
