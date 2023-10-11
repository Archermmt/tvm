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
 * \file src/contrib/msc/core/transform/fuse_tuple.cc
 * \brief Pass for fuse ShapeExpr.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../../../relax/transform/utils.h"

namespace tvm {
namespace relax {

/*!
 * \brief Fuse Tuple and TupleGetItem to BYOC
 */
class TupleFuser : public ExprMutator {
 public:
  explicit TupleFuser(IRModule ctx_module, const String& target, const String& entry_name)
      : ExprMutator(ctx_module) {
    mod_ = ctx_module;
    target_ = target;
    entry_name_ = entry_name;
  }

  IRModule Fuse() {
    GlobalVar main_var;
    for (const auto& [gv, func] : mod_->functions) {
      if (gv->name_hint == entry_name_) {
        main_var = gv;
        break;
      }
    }
    // update main
    ICHECK(main_var.defined()) << "Can not find entry func " << entry_name_;
    const auto& new_func = Downcast<Function>(VisitExpr(mod_->Lookup(entry_name_)));
    builder_->UpdateFunction(main_var, new_func);
    std::cout << "mod " << builder_->GetContextIRModule() << std::endl;
    return builder_->GetContextIRModule();
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) final {
    std::cout << "[TMINFO] find a tuple " << GetRef<Tuple>(val) << std::endl;
    ExprMutator::VisitBinding_(binding, val);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) final {
    std::cout << "[TMINFO] find a tuple getitem " << GetRef<TupleGetItem>(val) << std::endl;
    ExprMutator::VisitBinding_(binding, val);
  }

 private:
  IRModule mod_;
  String target_;
  String entry_name_;
};

IRModule FuseTuple(IRModule mod, const String& target, const String& entry_name) {
  return TupleFuser(mod, target, entry_name).Fuse();
}

namespace transform {

Pass FuseTuple(const String& target, const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::FuseTuple(m, target, entry_name); };
  return CreateModulePass(pass_func, 0, "FuseTuple", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FuseTuple").set_body_typed(FuseTuple);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
