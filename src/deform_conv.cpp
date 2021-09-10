// Copyright (c) OpenMMLab. All rights reserved
#include "deform_conv.h"

#include <cmath>
#include <vector>

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value) {
    OrtTensorTypeAndShapeInfo *info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

MMCVDeformConvKernel::MMCVDeformConvKernel(OrtApi api,
                                           const OrtKernelInfo *info)
    : api_(api), ort_(api_), info_(info) {
  std::vector<int64_t> stride =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "stride");
  stride_height_ = stride[0];
  stride_width_ = stride[1];
  std::vector<int64_t> padding =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "padding");
  padding_height_ = padding[0];
  padding_width_ = padding[1];
  std::vector<int64_t> dilation =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "dilation");
  dilation_height_ = dilation[0];
  dilation_width_ = dilation[1];
  deformable_group_ =
      ort_.KernelInfoGetAttribute<int64_t>(info, "deform_groups");
  group_ = ort_.KernelInfoGetAttribute<int64_t>(info, "groups");

  im2col_step_ = ort_.KernelInfoGetAttribute<int64_t>(info, "im2col_step");

  // create allocator
  allocator_ = Ort::AllocatorWithDefaultOptions();
}

void MMCVDeformConvKernel::Compute(OrtKernelContext *context) {
  const int64_t stride_height = stride_height_;
  const int64_t stride_width = stride_width_;
  const int64_t padding_height = padding_height_;
  const int64_t padding_width = padding_width_;
  const int64_t dilation_height = dilation_height_;
  const int64_t dilation_width = dilation_width_;
  const int64_t deformable_group = deformable_group_;
  const int64_t im2col_step = im2col_step_;
  const int64_t group = group_;

  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  const float *input_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(input));

  const OrtValue *offset = ort_.KernelContext_GetInput(context, 1);
  const float *offset_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(offset));

  const OrtValue *filter = ort_.KernelContext_GetInput(context, 2);
  const float *filter_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(filter));

  OrtTensorDimensions input_dims(ort_, input);
  OrtTensorDimensions filter_dims(ort_, filter);

  int64_t batch = input_dims[0];
  int64_t channels = input_dims[1];
  int64_t in_height = input_dims[2];
  int64_t in_width = input_dims[3];
  int64_t num_output = filter_dims[0];
  int64_t kernel_height = filter_dims[2];
  int64_t kernel_width = filter_dims[3];

  // get output memory
  int64_t out_height = floor((in_height + 2 * padding_height -
                              dilation_height * (kernel_height - 1) - 1) /
                                 stride_height +
                             1);
  int64_t out_width = floor(
      (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) /
          stride_width +
      1);

  std::vector<int64_t> output_dims = {batch / im2col_step, im2col_step,
                                      num_output, out_height, out_width};

  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, output_dims.data(), output_dims.size());
  float *out_ptr = ort_.GetTensorMutableData<float>(output);

  // allocate tmp memory
  int64_t column_len = (channels / group) * kernel_height * kernel_width *
                       out_height * out_width;
  float *columns = (float *)allocator_.Alloc(sizeof(float) * column_len);
}
