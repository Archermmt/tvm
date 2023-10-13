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
#include <vector>

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

  /*! \brief Get describe for default node input*/
  const String IdxInputBase(const MSCJoint& node, int idx = 0, bool as_raw = false) final {
    const auto& pair = node->ProducerAndIdxOf(idx);
    if (pair.first->optype == "input") {
      return "*" + IdxNodeBase(pair.first, as_raw);
    }
    if (pair.first->optype == "tuple" || pair.first->optype == "get_item") {
      return IdxNodeBase(pair.first, as_raw);
    }
    return "*" + IdxOutputBase(pair.first, pair.second, as_raw);
  }

  /*! \brief Get describe for default node output*/
  const String IdxOutputBase(const MSCJoint& node, int idx = 0, bool as_raw = false) final {
    if (node->optype == "argmax" || node->optype == "argmin") {
      ICHECK_EQ(idx, 0) << "argmax and argmin only has 1 output, get " << idx;
      return IdxNodeBase(node, as_raw) + "->getOutput(1)";
    }
    if (node->optype == "tuple") {
      return IdxNodeBase(node, as_raw) + "[" + std::to_string(idx) + "]";
    }
    if (node->optype == "get_item") {
      ICHECK_EQ(idx, 0) << "get item only has 1 output, get " << idx;
      return IdxNodeBase(node, as_raw);
    }
    return IdxNodeBase(node, as_raw) + "->getOutput(" + std::to_string(idx) + ")";
  }

  /*! \brief Get describe for default node weight*/
  const String IdxWeightBase(const MSCJoint& node, const String& wtype, bool as_raw = false) final {
    return "mWeights[\"" + node->WeightAt(wtype)->name + "\"]";
  }

 protected:
  /*! \brief Get the docs for the op*/
  const Array<Doc> GetOpCodes(const MSCJoint& node) final;

  /*! \brief Generate return on fail codes*/
  void ReturnOnFail(const String& flag, const String& err);

  /*! \brief Get the index tensor*/
  const String IdxTensor(const MSCTensor& tensor);

  /*! \brief Get the dtype from the datatype*/
  const String CppDType(const DataType& dtype);

  /*! \brief Generate describe for tensor bytes*/
  const String GetTensorBytes(const MSCTensor& tensor);

  /*! \brief Get the tensorrt dims from dims*/
  template <typename T>
  const String ToDims(const std::vector<T>& dims, bool use_ndim = true);
  const String ToDims(const Array<Integer>& dims, bool use_ndim = true);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_H_
