# Find the ONNX Runtime include directory and library.
#
# This module defines the `onnxruntime` imported target that encodes all
# necessary information in its target properties.

set(ONNXRUNTIME_PATH $ENV{HOME}/git/onnxruntime-linux-x64-1.8.1)

find_library(
  OnnxRuntime_LIBRARY
  NAMES onnxruntime
  PATH_SUFFIXES lib lib32 lib64
  PATHS ${ONNXRUNTIME_PATH}/lib
  DOC "The ONNXRuntime library")
  
if(NOT OnnxRuntime_LIBRARY)
  message(FATAL_ERROR "onnxruntime library not found")
endif()

find_path(
  OnnxRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_PATH}/include
  DOC "The ONNXRuntime include directory")
  
if(NOT OnnxRuntime_INCLUDE_DIR)
  message(FATAL_ERROR "onnxruntime includes not found")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OnnxRuntime
  REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR)

add_library(OnnxRuntime SHARED IMPORTED)
set_property(TARGET OnnxRuntime PROPERTY IMPORTED_LOCATION ${OnnxRuntime_LIBRARY})
set_property(TARGET OnnxRuntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_INCLUDE_DIR})

mark_as_advanced(OnnxRuntime_FOUND OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY)