// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

#include "MNIST.hpp"
#include "dcn.hpp"
#include "onnxruntime_cxx_api.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

int main(int argc, const char **argv) {

  std::string model_path = "";
  if (argc > 1) {
    for (int i = 1; i < argc; ++i)
      if (std::string_view{argv[i]} == "-f" && ++i < argc)
        model_path = argv[i];
  } else {
    cout << "To specify a map file use the following format: " << endl;
    cout << "Usage: [executable] [-f filename.onnx]" << endl;
  }

  // MNIST(model_path);
  // mDCN(model_path);
  DCN(model_path);

  printf("Done!\n");
  return 0;
}