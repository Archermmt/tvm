/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/contrib/msc/framework/tensorflow/tf_v1_opcode.cc
 */
#include "tf_v1_opcode.h"

#include <memory>
#include <string>

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> TFV1OpCode::GetDocs() {
  stack_.Config(this);
  CodeGenBuild();
  return stack_.GetDocs();
}

const std::pair<String, Array<String>> TFV1OpCode::GetPadding(const String& strides_key,
                                                              const String& kernel_key,
                                                              const String& padding_key) {
  String pad_mod = "";
  Array<String> padding;
  std::vector<int64_t> kernel_size;
  if (node()->optype == "nn.conv2d" || node()->optype == "msc.conv2d_bias") {
    const auto& weight = node()->WeightAt("weight");
    kernel_size.push_back(weight->DimAt("H")->value);
    kernel_size.push_back(weight->DimAt("W")->value);
  } else if (node()->optype == "nn.avg_pool2d" || node()->optype == "nn.max_pool2d") {
    ICHECK(node()->GetAttr(kernel_key, &kernel_size));
  } else {
    LOG_FATAL << "Unexpected padding node" << node();
  }
  const auto& strides = node()->GetTypeArrayAttr<int64_t>(strides_key);
  int64_t in_height = node()->InputAt(0)->DimAt("H")->value;
  int64_t in_width = node()->InputAt(0)->DimAt("W")->value;
  int64_t out_height = node()->OutputAt(0)->DimAt("H")->value;
  int64_t out_width = node()->OutputAt(0)->DimAt("W")->value;
  int64_t same_height = in_height / strides[0] + (in_height % strides[0] == 0 ? 0 : 1);
  int64_t same_width = in_width / strides[1] + (in_width % strides[1] == 0 ? 0 : 1);
  int64_t valid_height = (in_height - kernel_size[0] + 1) / strides[0];
  valid_height += (valid_height % strides[0] == 0 ? 0 : 1);
  int64_t valid_width = (in_width - kernel_size[1] + 1) / strides[1];
  valid_width += (valid_width % strides[1] == 0 ? 0 : 1);
  if (same_height == out_height && same_width == out_width) {
    pad_mod = "SAME";
  } else if (valid_height == out_height && valid_width == out_width) {
    pad_mod = "VALID";
  } else {
    const auto& src_padding = node()->GetTypeArrayAttr<int64_t>(padding_key);
    if (node()->optype == "nn.conv2d" || node()->optype == "msc.conv2d_bias" ||
        node()->optype == "nn.avg_pool2d" || node()->optype == "nn.max_pool2d") {
      const auto& out_layout = node()->GetTypeAttr<std::string>("out_layout");
      if (out_layout == "NHWC") {
        padding.push_back("[0, 0]");
      } else if (out_layout == "NCHW") {
        padding.push_back("[0, 0]");
        padding.push_back("[0, 0]");
      } else {
        LOG_FATAL << "Unexpected layout for padding node" << node();
      }
      if (src_padding.size() == 4) {
        padding.push_back("[" + std::to_string(src_padding[0]) + ", " +
                          std::to_string(src_padding[2]) + "]");
        padding.push_back("[" + std::to_string(src_padding[1]) + ", " +
                          std::to_string(src_padding[3]) + "]");
      } else {
        LOG_FATAL << "nn.conv2d/pool2d with unexpected padding " << node();
      }
      if (out_layout == "NHWC") {
        padding.push_back("[0, 0]");
      }
    } else {
      LOG_FATAL << "Unexpected padding node" << node();
    }
  }
  return std::make_pair(pad_mod, padding);
}

#define TFV1_OP_CODEGEN_METHODS(TypeName) \
 public:                                  \
  TypeName(const String& func_name) : TFV1OpCode(func_name) {}

class TFV1AstypeCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1AstypeCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_start()
        .op_input_arg()
        .call_dtype_arg(node()->OutputAt(0)->dtype)
        .op_name_arg()
        .op_end();
  }
};

class TFV1AxesCodeGen : public TFV1OpCode {
 public:
  TFV1AxesCodeGen(const String& func_name, const String& attr_name) : TFV1OpCode(func_name) {
    attr_name_ = attr_name;
  }

 protected:
  void CodeGenBuild() final {
    const String& key = node()->HasAttr("axes") ? "axes" : "axis";
    stack_.op_start().op_input_arg().op_list_arg<int>(key, attr_name_).op_name_arg().op_end();
  }

 private:
  String attr_name_;
};

class TFV1BroadcastToCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1BroadcastToCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_start().op_input_arg().op_list_arg<int>("shape").op_name_arg().op_end();
  }
};

class TFV1ConstantCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1ConstantCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_start()
        .call_str_arg(node()->name)
        .call_list_arg(node()->OutputAt(0)->shape)
        .call_str_arg(node()->OutputAt(0)->DTypeName())
        .call_arg("weights")
        .op_end();
  }
};

class TFV1ConvCodeGen : public TFV1OpCode {
 public:
  TFV1ConvCodeGen(const String& func_name, bool use_bias) : TFV1OpCode(func_name) {
    use_bias_ = use_bias;
  }

 protected:
  void CodeGenBuild() final {
    const auto& pair = GetPadding("strides");
    const auto& out_layout = node()->GetTypeAttr<std::string>("out_layout");
    int64_t groups = node()->GetTypeAttr<int64_t>("groups");
    std::vector<int> strides, dilation;
    const auto& attr_strides = node()->GetTypeArrayAttr<int>("strides");
    const auto& attr_dilation = node()->GetTypeArrayAttr<int>("dilation");
    if (out_layout == "NHWC") {
      strides = {1, attr_strides[0], attr_strides[1], 1};
      dilation = {1, attr_dilation[0], attr_dilation[1], 1};
    } else if (out_layout == "NCHW") {
      strides = {1, 1, attr_strides[0], attr_strides[1]};
      dilation = {1, 1, attr_dilation[0], attr_dilation[1]};
    } else {
      LOG_FATAL << "Unexpected layout for padding node" << node();
    }
    if (groups == 1) {
      stack_.op_start();
    } else if (groups == node()->InputAt(0)->DimAt("C")->value) {
      stack_.op_start("ops.nn_ops.depthwise_conv2d_native");
    } else {
      LOG_FATAL << "Unexpected conv with groups " << node();
    }
    stack_.op_input_arg()
        .op_weight_arg("weight")
        .call_list_arg(strides, "strides")
        .call_list_arg(dilation, "dilations")
        .op_str_arg("data_layout", "data_format");
    if (pair.first.size() > 0) {
      stack_.call_str_arg(pair.first, "padding");
    } else if (pair.second.size() > 0) {
      stack_.call_list_arg(pair.second, "padding");
    } else {
      LOG_FATAL << "Can not parse padding for " << node();
    }
    stack_.op_name_arg().op_end();
    if (use_bias_) {
      stack_.op_start("ops.nn_ops.bias_add")
          .op_output_arg()
          .op_weight_arg("bias")
          .call_str_arg(node()->name + "_bias", "name")
          .op_end();
    }
  }

 private:
  bool use_bias_;
};

class TFV1Pool2dCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1Pool2dCodeGen)

 protected:
  void CodeGenBuild() final {
    String pooling_type;
    if (node()->optype == "nn.avg_pool2d") {
      pooling_type = "AVG";
    } else if (node()->optype == "nn.max_pool2d") {
      pooling_type = "MAX";
    } else {
      LOG_FATAL << "Unexpected pool2d node " << node();
    }
    const auto& pair = GetPadding("strides", "pool_size");
    stack_.op_start()
        .op_input_arg()
        .op_list_arg<int>("pool_size", "window_shape")
        .call_str_arg(pooling_type, "pooling_type")
        .op_list_arg<int>("dilation", "dilation_rate")
        .op_list_arg<int>("strides");
    if (pair.first.size() > 0) {
      stack_.call_str_arg(pair.first, "padding");
    } else if (pair.second.size() > 0) {
      stack_.call_list_arg(pair.second, "padding");
    } else {
      LOG_FATAL << "Can not parse padding for " << node();
    }
    stack_.op_name_arg().op_end();
  }
};

class TFV1ReshapeCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1ReshapeCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_start().op_input_arg().op_list_arg<int>("shape").op_name_arg().op_end();
  }
};

class TensorflowSimpleCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TensorflowSimpleCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_start().op_inputs_arg(false).op_name_arg().op_end(); }
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TFV1OpCode>>> GetTFV1OpCodes() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<TFV1OpCode>>>();
  if (!map->empty()) return map;
  // binary && unary ops
  map->emplace("add", std::make_shared<TensorflowSimpleCodeGen>("tf_v1.add"));
  map->emplace("less", std::make_shared<TensorflowSimpleCodeGen>("tf_v1.less"));
  map->emplace("where", std::make_shared<TensorflowSimpleCodeGen>("tf_v1.where"));

  // create ops
  map->emplace("constant", std::make_shared<TFV1ConstantCodeGen>("get_variable"));

  // axis && axes ops
  map->emplace("expand_dims", std::make_shared<TFV1AxesCodeGen>("tf_v1.expand_dims", "axis"));
  map->emplace("squeeze", std::make_shared<TFV1AxesCodeGen>("ops.array_ops.squeeze", "axis"));

  // math ops
  map->emplace("astype", std::make_shared<TFV1AstypeCodeGen>("tf_v1.cast"));
  map->emplace("broadcast_to", std::make_shared<TFV1BroadcastToCodeGen>("tf_v1.broadcast_to"));
  map->emplace("reshape", std::make_shared<TFV1ReshapeCodeGen>("ops.array_ops.reshape"));

  // nn ops
  map->emplace("nn.avg_pool2d", std::make_shared<TFV1Pool2dCodeGen>("ops.nn_ops.pool"));
  map->emplace("nn.conv2d", std::make_shared<TFV1ConvCodeGen>("ops.nn_ops.conv2d", false));
  map->emplace("nn.max_pool2d", std::make_shared<TFV1Pool2dCodeGen>("ops.nn_ops.pool"));

  // msc ops
  map->emplace("msc.conv2d", std::make_shared<TFV1ConvCodeGen>("ops.nn_ops.conv2d", true));

  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
