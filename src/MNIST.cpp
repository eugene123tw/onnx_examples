// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using std::cout;
using std::vector;

void MNIST(std::string const model_path) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;

  printf("Using Onnxruntime C++ API\n");

  Ort::Session session(env, model_path.c_str(), session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  vector<const char *> input_node_names(num_input_nodes);
  vector<int64_t> input_node_dims; // simplify... this model has only 1
                                   // input node {1, 3, 224, 224}.
  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char *input_name = session.GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

  //*************************************************************************
  // Score the model using sample data, and inspect values
  // simplify ... using known dim values to calculate size
  size_t input_tensor_size = 224 * 224 * 3;
  // use OrtGetTensorShapeElementCount() to get official size!

  vector<float> input_tensor_values(input_tensor_size);
  vector<const char *> output_node_names = {"softmaxout_1"};

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_size,
      input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                  &input_tensor, 1, output_node_names.data(), 1);

  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float *floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);
}