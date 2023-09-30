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
 * \file src/contrib/msc/framework/tensorrt/transform_tensorrt.cc
 * \brief Pass for transform the function to tensorrt.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../../../relax/transform/utils.h"
#include "../../core/utils.h"

namespace tvm {
namespace relax {
using namespace tvm::contrib::msc;

Var EmitWithSuffix(BlockBuilder builder, const Call& call, const Var& src_var, const Span& src_span,
                   const String& suffix) {
  const auto& src_name = SpanUtils::GetAttr(src_span, "name");
  call->span = SpanUtils::SetAttr(call->span, "name", src_name + "_" + suffix);
  return builder->Emit(call, src_var->name_hint() + "_" + suffix);
}

using FRewriteTensorRT = runtime::TypedPackedFunc<Expr(
    BlockBuilder builder, const Var& var, const Call& call, const Map<Expr, Call>& new_calls)>;

Expr RewriteElemwise(BlockBuilder builder, const Var& var, const Call& call,
                     const Map<Expr, Call>& new_calls) {
  Array<PrimExpr> empty;
  const auto& shape_a =
      Downcast<TensorStructInfo>(GetStructInfo(call->args[0]))->GetShape().value_or(empty);
  const auto& shape_b =
      Downcast<TensorStructInfo>(GetStructInfo(call->args[1]))->GetShape().value_or(empty);
  static const Op& reshape_op = Op::Get("relax.reshape");
  if (shape_a.size() > shape_b.size()) {
    ICHECK(shape_b.size() == 1) << "broadcast only support 1 dim, get " << shape_b;
    Array<PrimExpr> exp_shape(shape_a.size(), Integer(1));
    exp_shape.Set(shape_a.size() - 1, shape_b[0]);
    const auto& expand_b = Call(reshape_op, {call->args[1], ShapeExpr(exp_shape)}, Attrs(), {});
    const auto& expand_b_var = EmitWithSuffix(builder, expand_b, var, call->span, "expand_b");
    return Call(call->op, {call->args[0], expand_b_var}, call->attrs, call->sinfo_args, call->span);
  }
  if (shape_a.size() < shape_b.size()) {
    ICHECK(shape_a.size() == 1) << "broadcast only support 1 dim, get " << shape_a;
    Array<PrimExpr> exp_shape(shape_a.size(), Integer(1));
    exp_shape.Set(shape_b.size() - 1, shape_a[0]);
    const auto& expand_a = Call(reshape_op, {call->args[0], ShapeExpr(exp_shape)}, Attrs(), {});
    const auto& expand_a_var = EmitWithSuffix(builder, expand_a, var, call->span, "expand_a");
    return Call(call->op, {expand_a_var, call->args[1]}, call->attrs, call->sinfo_args, call->span);
  }
  return call;
}

Expr RewriteAdd(BlockBuilder builder, const Var& var, const Call& call,
                const Map<Expr, Call>& new_calls) {
  if (new_calls.count(call->args[0]) &&
      new_calls[call->args[0]]->op == Op::Get("relax.nn.conv1d")) {
    const auto& reshape = Downcast<Call>(builder->LookupBinding(Downcast<Var>(call->args[0])));
    if (reshape->op != Op::Get("relax.reshape")) {
      return call;
    }
    const auto& conv2d = Downcast<Call>(builder->LookupBinding(Downcast<Var>(reshape->args[0])));
    if (conv2d->op != Op::Get("relax.nn.conv2d")) {
      return call;
    }
    Array<PrimExpr> empty;
    const auto& input_shape =
        Downcast<TensorStructInfo>(GetStructInfo(call->args[0]))->GetShape().value_or(empty);
    const auto& bias_shape =
        Downcast<TensorStructInfo>(GetStructInfo(call->args[1]))->GetShape().value_or(empty);
    if (input_shape.size() == 0 || bias_shape.size() == 0) {
      return call;
    }
    const auto* conv_attrs = conv2d->attrs.as<Conv2DAttrs>();
    if (conv_attrs->data_layout == "NCHW") {
      // expand bias reshape
      Array<PrimExpr> exp_bias_shape{bias_shape[0], bias_shape[1], Integer(1), bias_shape[2]};
      static const Op& reshape_op = Op::Get("relax.reshape");
      const auto& exp_bias =
          Call(reshape_op, {call->args[1], ShapeExpr(exp_bias_shape)}, Attrs(), {});
      const auto& exp_bias_var = EmitWithSuffix(builder, exp_bias, var, call->span, "exp_bias");
      // redirect to conv2d
      static const Op& add_op = Op::Get("relax.add");
      const auto& exp_add = Call(add_op, {reshape->args[0], exp_bias_var}, Attrs(), {});
      const auto& exp_add_var = EmitWithSuffix(builder, exp_add, var, call->span, "exp");
      // reduce output
      return Call(reshape_op, {exp_add_var, ShapeExpr(input_shape)}, Attrs(), {}, call->span);
    } else {
      LOG_FATAL << "Unexpected data layout " << conv_attrs->data_layout;
    }
  }
  return RewriteElemwise(builder, var, call, new_calls);
}

Expr RewriteConv1d(BlockBuilder builder, const Var& var, const Call& call,
                   const Map<Expr, Call>& new_calls) {
  const auto* src_attrs = call->attrs.as<Conv1DAttrs>();
  Array<PrimExpr> empty;
  const auto& input_shape =
      Downcast<TensorStructInfo>(GetStructInfo(call->args[0]))->GetShape().value_or(empty);
  const auto& weight_shape =
      Downcast<TensorStructInfo>(GetStructInfo(call->args[1]))->GetShape().value_or(empty);
  const auto& output_shape =
      Downcast<TensorStructInfo>(GetStructInfo(var))->GetShape().value_or(empty);
  if (input_shape.size() == 0 || weight_shape.size() == 0 || output_shape.size() == 0) {
    return call;
  }
  if (src_attrs->data_layout == "NCW") {
    Array<Expr> new_args;
    // expand inputs
    Array<PrimExpr> exp_input_shape{input_shape[0], input_shape[1], Integer(1), input_shape[2]};
    Array<PrimExpr> exp_weight_shape{weight_shape[0], weight_shape[1], Integer(1), weight_shape[2]};
    static const Op& reshape_op = Op::Get("relax.reshape");
    const auto& exp_input =
        Call(reshape_op, {call->args[0], ShapeExpr(exp_input_shape)}, Attrs(), {});
    new_args.push_back(EmitWithSuffix(builder, exp_input, var, call->span, "exp_input"));
    const auto& exp_weight =
        Call(reshape_op, {call->args[1], ShapeExpr(exp_weight_shape)}, Attrs(), {});
    new_args.push_back(EmitWithSuffix(builder, exp_weight, var, call->span, "exp_weight"));
    // change to conv2d
    static const Op& conv2d_op = Op::Get("relax.nn.conv2d");
    auto conv_attrs = make_object<Conv2DAttrs>();
    conv_attrs->strides = Array<IntImm>{src_attrs->strides[0], Integer(1)};
    conv_attrs->padding =
        Array<IntImm>{Integer(0), src_attrs->padding[0], Integer(0), src_attrs->padding[1]};
    conv_attrs->dilation = Array<IntImm>{src_attrs->dilation[0], Integer(1)};
    conv_attrs->groups = src_attrs->groups;
    conv_attrs->data_layout = "NCHW";
    conv_attrs->kernel_layout = "OIHW";
    conv_attrs->out_layout = "NCHW";
    conv_attrs->out_dtype = src_attrs->out_dtype;
    const auto& conv2d = Call(conv2d_op, new_args, Attrs(conv_attrs), {});
    const auto& conv2d_var = EmitWithSuffix(builder, conv2d, var, call->span, "exp");
    // reduce output
    return Call(reshape_op, {conv2d_var, ShapeExpr(output_shape)}, Attrs(), {}, call->span);
  } else {
    LOG_FATAL << "Unexpected data layout " << src_attrs->data_layout;
  }
  return call;
}

// nn ops
TVM_REGISTER_OP("relax.nn.conv1d").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteConv1d);

// math ops
TVM_REGISTER_OP("relax.add").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteAdd);
TVM_REGISTER_OP("relax.subtract").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.multiply").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);
TVM_REGISTER_OP("relax.divide").set_attr<FRewriteTensorRT>("FRewriteTensorRT", RewriteElemwise);

class TensorRTTransformer : public ExprMutator {
 public:
  explicit TensorRTTransformer(IRModule ctx_module) : ExprMutator(ctx_module) {}

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    if (const auto* op_node = call_node->op.as<OpNode>()) {
      const auto& op = Downcast<Op>(GetRef<Op>(op_node));
      const auto& rewrite_map = Op::GetAttrMap<FRewriteTensorRT>("FRewriteTensorRT");
      if (rewrite_map.count(op)) {
        const auto& call = GetRef<Call>(call_node);
        FRewriteTensorRT f = rewrite_map[op];
        const auto& new_call = f(builder_, binding->var, call, new_calls_);
        if (new_call != call) {
          ReEmitBinding(binding, builder_->Normalize(new_call));
          new_calls_.Set(binding->var, call);
        }
      }
    }
    if (!new_calls_.count(binding->var)) {
      ExprMutator::VisitBinding_(binding, call_node);
    }
  }

 private:
  Map<Expr, Call> new_calls_;
};

Function TransformTensorRT(const Function& func, const IRModule& module) {
  return Downcast<Function>(TensorRTTransformer(module).VisitExpr(func));
}

namespace transform {

Pass TransformTensorRT() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return relax::TransformTensorRT(f, m); };
  return CreateFunctionPass(pass_func, 0, "TransformTensorRT", {});
}

TVM_REGISTER_GLOBAL("relax.transform.TransformTensorRT").set_body_typed(TransformTensorRT);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
