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
 * \file src/contrib/msc/framework/tensorrt/codegen.cc
 * \brief Codegen related classes.
 */

#include "codegen.h"

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::relax;

void TensorRTCodeGen::CodeGenClassDeclare() {
  stack_.line("#include \"NvInfer.h\"")
      .line("#include \"NvInferRuntimeCommon.h\"")
      .line("#include \"utils/base.h\"")
      .line()
      .line("using namsespace nvinfer1;")
      .line();
  StartNamespace();
  stack_.class_def(graph()->name)
      .class_start()
      .scope_start("public:")
      .func_def("build", "bool")
      .func_arg("inputs", "ITensor**")
      .func_arg("outputs", "ITensor**")
      .func_arg("builder", "TRTUniquePtr<IBuilder>&")
      .func_arg("network", "TRTUniquePtr<INetworkDefinition>&");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.func_arg("config", "TRTUniquePtr<IBuilderConfig>&");
  }
  stack_.func_arg("logger", "TRTLogger&")
      .func_start()
      .func_end()
      .func_def("clean_up", "bool")
      .func_start()
      .func_end()
      .scope_end();
  stack_.class_end();
  EndNamespace();
}

void TensorRTCodeGen::CodeGenClassDefine() {}

void TensorRTCodeGen::CodeGenMain() {}

void TensorRTCodeGen::CodeGenCmake() {}

const Array<Doc> TensorRTCodeGen::GetOpCodes(const MSCJoint& node) {
  const auto& ops_map = GetTensorRTOpCodes();
  auto it = ops_map->find(node->optype);
  ICHECK(it != ops_map->end()) << "Unsupported tensorrt op(" << node->optype << "): " << node;
  it->second->Config(node, config());
  try {
    return it->second->GetDocs();
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to get docs for " << node << " : " << err.message();
    throw err;
  }
}

TVM_REGISTER_GLOBAL("msc.framework.tensorrt.GetTensorRTSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String print_config) -> Map<String, String> {
      TensorRTCodeGen codegen = TensorRTCodeGen(graph, codegen_config);
      return codegen.GetSources(print_config);
    });

/*!
 * \brief Create runtime modules for TensorRT.
 * \param functions The extern functions to be compiled via TensorRT
 * \return Runtime modules.
 */
Array<runtime::Module> TensorRTCompiler(Array<Function> functions,
                                        Map<String, ObjectRef> /*unused*/,
                                        Map<Constant, String> constant_names) {
  Array<runtime::Module> compiled_functions;
  for (const auto& func : functions) {
    std::cout << "[TMINFO] processing " << func << std::endl;
  }
  return compiled_functions;
}

TVM_REGISTER_GLOBAL("relax.ext.msc_tensorrt").set_body_typed(TensorRTCompiler);

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
