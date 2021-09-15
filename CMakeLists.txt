cmake_minimum_required(VERSION 3.11.3)

# Set the C++ standard we will use
set(CMAKE_CXX_STANDARD 17)

# Add the path of the cmake files to the CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# Add LibTorch to CMAKE PREFIX PATH so find_package knows where it is
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /home/yuchunli/git/libtorch-shared-with-deps-1.8.2+cpu/libtorch)

project(ONNX_CPP_EXAMPLES)

find_package(OnnxRuntime)
find_package(Torch REQUIRED)

# Add the testing executable
add_executable(
  main 
  src/main.cpp 
  src/MNIST.cpp 
  src/dcn.cpp 
  src/modulated_deform_conv.cpp
  src/deform_conv.cpp
)
target_link_libraries(main OnnxRuntime ${TORCH_LIBRARIES})

add_executable(simple_torch src/simple_torch.cpp)
target_link_libraries(simple_torch ${TORCH_LIBRARIES})