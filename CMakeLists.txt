cmake_minimum_required(VERSION 3.11.3)

# Set the C++ standard we will use
set(CMAKE_CXX_STANDARD 17)

# Add the path of the cmake files to the CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

project(ONNX_CPP_EXAMPLES)

find_package(OnnxRuntime)

# Add the testing executable
add_executable(main src/main.cpp src/MNIST.cpp src/dcn.cpp src/onnxruntime_register.cpp src/modulated_deform_conv.cpp)
target_link_libraries(main OnnxRuntime)