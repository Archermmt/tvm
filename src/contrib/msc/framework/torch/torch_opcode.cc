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
 * \file src/contrib/msc/framework/torch/torch_opcode.cc
 */
#include "torch_opcode.h"

#include <memory>
#include <string>

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> TorchOpCode::GetDocs() {
  stack_.Config(this);
  if (is_init_) {
    CodeGenInit();
  } else {
    CodeGenForward();
  }
  return stack_.GetDocs();
}

void TorchOpCode::CodeGenInit() { stack_.comment(node()->name + "(" + node()->optype + ")"); }

void TorchOpCode::CodeGenForward() { stack_.op_start().op_inputs_arg(false).op_end(); }

const std::vector<int> TorchOpCode::GetPadding(const String& key) {
  std::vector<int> padding, src_padding;
  ICHECK(node()->GetAttr(key, &src_padding));
  if (node()->optype == "nn.conv1d") {
    if (src_padding.size() == 2) {
      ICHECK(src_padding[0] == src_padding[1]) << "Only accept symmetric padding, get " << node();
      padding.push_back(src_padding[0]);
    } else {
      LOG_FATAL << "nn.conv1d with unexpected padding " << node();
    }
  }
  return padding;
}

#define TORCH_OP_CODEGEN_METHODS(TypeName)                     \
 public:                                                       \
  TypeName(const String& module_name, const String& func_name) \
      : TorchOpCode(module_name, func_name) {}

class TorchConvCodeGen : public TorchOpCode {
 public:
  TorchConvCodeGen(const String& module_name, const String& func_name, bool use_bias)
      : TorchOpCode(module_name, func_name), use_bias_(use_bias) {}

 protected:
  void CodeGenInit() final {
    const auto& weight = node()->WeightAt("weight");
    std::vector<int64_t> kernel_size;
    for (size_t i = 0; i < weight->Ndim(); i++) {
      if (weight->layout[i].name() == "I" || weight->layout[i].name() == "O") {
        continue;
      }
      kernel_size.push_back(weight->DimAt(i)->value);
    }
    stack_.op_start()
        .call_arg(weight->DimAt("I"), "in_channels")
        .call_arg(weight->DimAt("O"), "out_channels")
        .call_list_arg(kernel_size, "kernel_size")
        .op_list_arg<int>("strides", "stride")
        .call_list_arg(GetPadding())
        .op_list_arg<int>("dilation")
        .op_arg<int>("groups")
        .call_arg(use_bias_, "bias")
        .op_end();
  }

 private:
  bool use_bias_;
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TorchOpCode>>> GetTorchOpCodes() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<TorchOpCode>>>();
  if (!map->empty()) return map;

  // msc ops
  map->emplace("msc.conv1d_bias",
               std::make_shared<TorchConvCodeGen>("nn.Conv1d", "functional.conv1d", true));
  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
