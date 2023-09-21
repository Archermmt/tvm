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
 * \file src/contrib/msc/framework/tensorrt/tensorrt_opcode.cc
 */
#include "tensorrt_opcode.h"

#include <memory>
#include <string>

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> TensorRTOpCode::GetDocs() {
  stack_.Config(this);
  CodeGenBuild();
  return stack_.GetDocs();
}

const String TensorRTOpCode::DType(const DataType& dtype) {
  const String& dtype_name = BaseOpCode<TensorRTCodeGenConfig>::DType(dtype);
  String dtype_enum;
  if (dtype_name == "int8") {
    dtype_enum = "DataType::kINT8";
  } else if (dtype_name == "float16") {
    dtype_enum = "DataType::kHALF";
  } else if (dtype_name == "float32") {
    dtype_enum = "DataType::kFLOAT";
  } else {
    LOG_FATAL << "Unexpected dtype for TensorRT " << dtype_name;
  }
  return dtype_enum;
}

template <typename T>
const String TensorRTOpCode::ToDims(const std::vector<T>& dims, bool use_ndim) {
  if (dims.size() == 2 && !use_ndim) {
    return "DimsHW{" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "}";
  }
  String dims_str = "Dims({" + std::to_string(dims.size()) + ",{";
  for (size_t i = 0; i < dims.size(); i++) {
    dims_str = dims_str + std::to_string(dims[i]) + (i < dims.size() - 1 ? "," : "");
  }
  dims_str = dims_str + "}})";
  return dims_str;
}

const String TensorRTOpCode::ToDims(const Array<Integer>& dims, bool use_ndim) {
  std::vector<int64_t> int_dims;
  for (const auto& d : dims) {
    int_dims.push_back(d->value);
  }
  return ToDims(int_dims, use_ndim);
}

const String TensorRTOpCode::AttrToDims(const String& key, bool use_ndim) {
  const auto& dims = node()->GetTypeArrayAttr<int64_t>(key);
  return ToDims(dims, use_ndim);
}

template <typename T>
void TensorRTOpCode::SetLayerByAttr(const String& method, const String& key) {
  stack_.call_start(IdxNode() + "->set" + method).op_arg<T>(key).call_end();
}

template <typename T>
void TensorRTOpCode::SetLayerByValue(const String& method, const T& value) {
  stack_.call_start(IdxNode() + "->set" + method).call_arg(value).call_end();
}

template <typename T>
void TensorRTOpCode::SetLayerByDimsAttr(const String& method, const String& key, bool use_ndim) {
  stack_.call_start(IdxNode() + "->set" + method).call_arg(AttrToDims(key, use_ndim)).call_end();
}

template <typename T>
void TensorRTOpCode::SetLayerByDimsValue(const String& method, const std::vector<T>& value,
                                         bool use_ndim) {
  stack_.call_start(IdxNode() + "->set" + method).call_arg(ToDims(value, use_ndim)).call_end();
}

void TensorRTOpCode::SetLayerByDimsValue(const String& method, const Array<Integer>& value,
                                         bool use_ndim) {
  stack_.call_start(IdxNode() + "->set" + method).call_arg(ToDims(value, use_ndim)).call_end();
}

#define TENSORRT_OP_CODEGEN_METHODS(TypeName) \
 public:                                      \
  TypeName(const String& func_name) : TensorRTOpCode(func_name) {}

class TensorRTConvCodeGen : public TensorRTOpCode {
 public:
  TensorRTConvCodeGen(const String& func_name, bool use_bias) : TensorRTOpCode(func_name) {
    use_bias_ = use_bias;
  }

 protected:
  void CodeGenBuild() final {
    const auto& weight = node()->WeightAt("weight");
    std::vector<int64_t> kernel_size;
    for (size_t i = 0; i < weight->Ndim(); i++) {
      if (weight->layout[i].name() == "I" || weight->layout[i].name() == "O") {
        continue;
      }
      kernel_size.push_back(weight->DimAt(i)->value);
    }
    stack_.op_start()
        .op_input_arg()
        .call_arg(weight->DimAt("O"))
        .call_arg(ToDims(kernel_size, false))
        .op_weight_arg("weight");
    if (use_bias_) {
      stack_.op_weight_arg("bias");
    } else {
      stack_.call_arg("mWeights[\"" + node()->name + ".bias\"]");
    }
    stack_.op_end();
    SetLayerByDimsAttr<int>("Stride", "strides", false);
    SetLayerByDimsAttr<int>("Dilation", "dilation", false);
    SetLayerByAttr<int>("NbGroups", "groups");
  }

 private:
  bool use_bias_;
};

class TensorRTInputCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTInputCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& output = node()->OutputAt(0);
    stack_.op_start()
        .call_arg(DocUtils::ToStrDoc(output->alias))
        .call_dtype_arg(output->dtype)
        .call_arg(ToDims(output->shape))
        .op_end();
  }
};

class TensorRTReshapeCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTReshapeCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& output = node()->OutputAt(0);
    stack_.op_start().op_input_arg().op_end();
    SetLayerByDimsValue("ReshapeDimensions", output->shape);
  }
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TensorRTOpCode>>>
GetTensorRTOpCodes() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<TensorRTOpCode>>>();
  if (!map->empty()) return map;

  // math ops
  map->emplace("reshape", std::make_shared<TensorRTReshapeCodeGen>("addShuffle"));

  // nn ops
  map->emplace("nn.conv2d", std::make_shared<TensorRTConvCodeGen>("addConvolutionNd", false));

  // special op
  map->emplace("input", std::make_shared<TensorRTInputCodeGen>("addInput"));

  // msc ops
  map->emplace("msc.conv2d_bias", std::make_shared<TensorRTConvCodeGen>("addConvolutionNd", true));

  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
