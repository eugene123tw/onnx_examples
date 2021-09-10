#ifndef ONNXRUNTIME_DCN
#define ONNXRUNTIME_DCN

#include "onnxruntime_cxx_api.h"
void get_input_name(Ort::Session const &session);
void mDCN(std::string model_path);

#endif