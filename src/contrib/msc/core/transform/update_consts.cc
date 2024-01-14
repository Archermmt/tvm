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
 * \file src/contrib/msc/core/transform/update_consts.cc
 * \brief Pass for fuse ShapeExpr.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../../../relax/transform/utils.h"
#include "../utils.h"

namespace tvm {
namespace relax {

using namespace tvm::contrib::msc;

/*!
 * \brief Reset NDArray to Constant
 */
class ConstResetter : public ExprMutator {
 public:
  explicit ConstResetter(const Map<String, tvm::runtime::NDArray>& datas) : datas_(datas) {}

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const ConstantNode* op) final {
    const auto& name = SpanUtils::GetAttr(op->span, msc_attr::kName);
    if (datas_.count(name)) {
      return Constant(datas_[name], Downcast<StructInfo>(op->struct_info_), op->span);
    }
    return ExprMutator::VisitExpr_(op);
  }

  Map<String, tvm::runtime::NDArray> datas_;
};

IRModule UpdateConsts(IRModule mod, const Map<String, tvm::runtime::NDArray>& datas) {
  IRModuleNode* new_module = mod.CopyOnWrite();
  for (const auto& [gv, func] : mod->functions) {
    if (func->IsInstance<FunctionNode>()) {
      Expr new_func = ConstResetter(datas).VisitExpr(func);
      new_module->Update(gv, Downcast<Function>(new_func));
    } else {
      new_module->Update(gv, func);
    }
  }
  return GetRef<IRModule>(new_module);
}

namespace transform {

Pass UpdateConsts(const Map<String, tvm::runtime::NDArray>& datas) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::UpdateConsts(m, datas); };
  return CreateModulePass(pass_func, 0, "UpdateConsts", {});
}

TVM_REGISTER_GLOBAL("relax.transform.UpdateConsts").set_body_typed(UpdateConsts);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
