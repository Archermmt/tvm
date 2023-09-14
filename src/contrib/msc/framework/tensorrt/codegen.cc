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

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::relax;

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
