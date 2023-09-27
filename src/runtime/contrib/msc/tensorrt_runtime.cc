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
 * \file src/runtime/contrib/tensorrt/tensorrt_runtime.cc
 * \brief JSON runtime implementation for TensorRT.
 */

#include <dmlc/parameter.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

class MSCTensorRTRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The MSC TensorRT runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit MSCTensorRTRuntime(const std::string& symbol_name, const std::string& graph_json,
                              const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  ~MSCTensorRTRuntime() override { VLOG(1) << "Destroying MSC TensorRT runtime"; }

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const final { return "msc_tensorrt"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  /*!
   * \brief Initialize runtime.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    LoadGlobalOptions();
    std::cout << "engine_file_ " << engine_file_ << std::endl;
  }

  void LoadGlobalOptions() {
    // These settings are global to the entire subgraph. Codegen will add them as attributes to all
    // op nodes. Read from first one.
    for (size_t i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i].HasAttr("msc_global_options_num")) {
        engine_file_ = nodes_[i].GetAttr<std::vector<std::string>>("msc_global_engine")[0];
      }
    }
  }

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
  void LoadEngine(const String& engine_file) {
    std::cout << "[TMINFO] get engine_file " << engine_file << std::endl;
  }

  void Run() override { std::cout << "[TMINFO] calling run of msc tensorrt" << std::endl; }

#else   // TVM_GRAPH_EXECUTOR_TENSORRT
  void LoadEngine(const String& engine_file) {
    LOG(FATAL) << "TensorRT runtime is not enabled. "
               << "Please build with USE_TENSORRT_RUNTIME.";
  }

  void Run() override {
    LOG(FATAL) << "TensorRT runtime is not enabled. "
               << "Please build with USE_TENSORRT_RUNTIME.";
  }
#endif  // TVM_GRAPH_EXECUTOR_TENSORRT

 private:
  String engine_file_;
};

runtime::Module MSCTensorRTRuntimeCreate(const String& symbol_name, const String& graph_json,
                                         const Array<String>& const_names) {
  auto n = make_object<MSCTensorRTRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.msc_tensorrt_runtime_create").set_body_typed(MSCTensorRTRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_msc_tensorrt")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<MSCTensorRTRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
