#include "modulated_deform_conv.h"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::vector;

void get_input_name(Ort::Session const &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t num_input_nodes = session.GetInputCount();
  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char *input_name = session.GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    printf("Input %d : num_dims=%zu\n", i, tensor_info.GetShape().size());
    for (int j = 0; j < tensor_info.GetShape().size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, tensor_info.GetShape()[j]);
  }
}

void mDCN(std::string model_path) {
  vector<float> input{
      1., 2., 3., 0., 1., 2., 3., 5., 2.,
  };
  vector<int64_t> input_shape{1, 1, 3, 3};
  vector<float> offset{1.7000, 2.9000, 3.4000, 4.8000, 1.1000, 2.0000, 2.1000,
                       1.9000, 3.1000, 5.1000, 5.9000, 4.9000, 2.0000, 4.1000,
                       4.0000, 6.6000, 1.6000, 2.7000, 3.8000, 3.1000, 2.5000,
                       4.3000, 4.2000, 5.3000, 1.7000, 3.3000, 3.6000, 4.5000,
                       1.7000, 3.4000, 5.2000, 6.1000};
  vector<int64_t> offset_shape{1, 8, 2, 2};

  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_tensor =
      Ort::Value::CreateTensor<float>(mem_info, input.data(), input.size(),
                                      input_shape.data(), input_shape.size());

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  MMCVModulatedDeformConvOp mdcn_op{};

  Ort::CustomOpDomain custom_op_domain("mmcv");
  custom_op_domain.Add(&mdcn_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);
  Ort::Session session(env, model_path.c_str(), session_options);
  get_input_name(session);
}