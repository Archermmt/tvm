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

#define TENSORRT_OP_CODEGEN_METHODS(TypeName) \
 public:                                      \
  TypeName(const String& func_name) : TensorRTOpCode(func_name) {}

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TensorRTOpCode>>>
GetTensorRTOpCodes() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<TensorRTOpCode>>>();
  if (!map->empty()) return map;

  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
