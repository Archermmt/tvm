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
 * \file src/contrib/msc/framework/tensorrt/codegen.h
 * \brief Relax codegen for MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_H_

#include <string>

#include "../../core/codegen/base_codegen.h"
#include "../../core/codegen/cpp_codegen.h"
#include "config.h"
#include "tensorrt_opcode.h"

namespace tvm {
namespace contrib {
namespace msc {

class TensorRTCodeGen : public CppCodeGen<TensorRTCodeGenConfig> {
 public:
  /*!
   * \brief The constructor of TensorRTCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit TensorRTCodeGen(const MSCGraph& graph, const std::string& config = "")
      : CppCodeGen<TensorRTCodeGenConfig>(graph, config) {}

  /*! \brief Stack the docs for the class declare*/
  void CodeGenClassDeclare() final;

  /*! \brief Stack the docs for the class define*/
  void CodeGenClassDefine() final;

  /*! \brief Stack the docs for the main func*/
  void CodeGenMain() final;

  /*! \brief Stack the docs for the class define*/
  void CodeGenCmake() final;

 protected:
  /*! \brief Get the docs for the op*/
  const Array<Doc> GetOpCodes(const MSCJoint& node) final;

  /*! \brief Generate return on fail codes*/
  void ReturnOnFail(const String& flag, const String& err);

  /*! \brief Get the dtype from the datatype*/
  const String CppDType(const DataType& dtype);

  /*! \brief Generate describe for tensor bytes*/
  const String GetTensorBytes(const MSCTensor& tensor);

  /*! \brief Generate assign_to for func call*/
  const AssignDoc GetAssignTo(const String& assign_to, const String& type);

  /*! \brief Generate caller for func call*/
  const PtrAttrAccessDoc GetPtrCaller(const String& caller);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_H_
