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
 * \file src/contrib/msc/plugin/tvm_codegen.cc
 */
#include "tvm_codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

void TVMPluginCodeGen::CodeGenAttrDeclare(const Plugin& plugin) {
  BasePluginCodeGen<TVMPluginCodeGenConfig>::CodeGenAttrDeclare(plugin);
  /*
  const auto& attr_name = MetaAttrCls(plugin);
  const auto& r_attr_name = RelaxMetaAttrCls(plugin);
  stack_.struct_start(r_attr_name + " : public tvm::AttrsNode<" + r_attr_name + ">");
  // define attributes
  stack_.comment("define attributes");
  for (const auto& a : plugin->attrs) {
    stack_.declare(ToTVMType(a->type), a->name);
    if (a->default_value.size() > 0) {
      stack_.declare_arg(a->default_value);
    }
  }
  stack_.line()
      .comment("register attributes")
      .func_def("TVM_DECLARE_ATTRS", DocSymbol::Empty())
      .func_arg(r_attr_name, DocSymbol::Empty())
      .func_arg("\"relax.attrs." + r_attr_name + "\"", DocSymbol::Empty())
      .func_start();
  for (const auto& a : plugin->attrs) {
    stack_.func_call("TVM_ATTR_FIELD")
        .call_arg(a->name)
        .method_call("describe")
        .call_arg(DocUtils::ToStrDoc(a->describe));
  }
  stack_.func_end().struct_end();
  // convert method to meta
  stack_.comment("convert relax attrs to meta method")
      .func_def(plugin->name + "_attrs_to_meta", attr_name)
      .func_arg("attrs", "const " + r_attr_name + "*");
  // convert method to meta in runtime
  stack_.comment("convert arguments to meta method")
      .func_def(plugin->name + "_args_to_meta", attr_name);
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, ToTVMType(a->type));
  }
  */
  const auto& attr_name = MetaAttrCls(plugin);
  // deserialize method for attr
  stack_.comment("deserialize method").func_def(attr_name + "_deserialize", "const " + attr_name);
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, "const StringImm&");
  }
}

void TVMPluginCodeGen::CodeGenAttrDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  // const auto& r_attr_name = RelaxMetaAttrCls(plugin);
  //  register attrs
  // stack_.func_call("TVM_REGISTER_NODE_TYPE").call_arg(RelaxMetaAttrCls(plugin)).line();
  //  deserialize method for attr
  stack_.func_def(attr_name + "_deserialize", "const " + attr_name);
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, "const StringImm&");
  }
  stack_.func_start().declare(attr_name, "meta_attrs");
  for (const auto& a : plugin->attrs) {
    stack_.func_call("SerializeUtils::FromString")
        .call_arg(DocUtils::ToAttrAccessDoc(DocUtils::ToPtrDoc(a->name), "value"))
        .call_arg(DocUtils::ToAttrAccessDoc("meta_attrs", a->name));
  }
  stack_.func_end("meta_attrs");

  /*
  auto assign_attr = [this](const PluginAttr& attr, const String& src_attr) {
    const auto& meta_attr_doc = DocUtils::ToAttrAccessDoc("meta_attrs", attr->name);
    if (IsListType(attr->type)) {
      stack_.for_start("v", src_attr)
          .func_call("push_back", NullOpt, meta_attr_doc)
          .call_arg(DocUtils::ToAttrAccessDoc(DocUtils::ToPtrDoc("v"), "value"))
          .for_end();
    } else if (attr->type == "string") {
      stack_.func_call("std::string", DocUtils::ToDeclareDoc("", meta_attr_doc)).call_arg(src_attr);
    } else {
      stack_.assign(meta_attr_doc,
                    DocUtils::ToAttrAccessDoc(DocUtils::ToPtrDoc(src_attr), "value"));
    }
  };
  // register attrs
  stack_.func_call("TVM_REGISTER_NODE_TYPE").call_arg(RelaxMetaAttrCls(plugin)).line();
  // convert method to meta
  stack_.func_def(plugin->name + "_attrs_to_meta", attr_name)
      .func_arg("attrs", "const " + r_attr_name + "*")
      .func_start()
      .declare(attr_name, "meta_attrs");
  for (const auto& a : plugin->attrs) {
    assign_attr(a, "attrs->" + a->name);
  }
  stack_.func_end("meta_attrs");
  // convert method to meta in runtime
  stack_.func_def(plugin->name + "_args_to_meta", attr_name);
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, ToTVMType(a->type));
  }
  stack_.func_start().declare(attr_name, "meta_attrs");
  for (const auto& a : plugin->attrs) {
    assign_attr(a, a->name);
  }
  stack_.func_end("meta_attrs");
  */
}

void TVMPluginCodeGen::CodeGenOpDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  // const auto& r_attr_name = RelaxMetaAttrCls(plugin);
  // infer struct info
  Array<String> infer_args{"input_metas", "meta_attrs", "false"};
  stack_.func_def("InferStructInfo" + plugin->name, "Array<TensorStructInfo>");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "const Expr&");
  }
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, "const StringImm&");
  }
  stack_.func_start()
      .comment("extract meta attrs")
      .func_call(attr_name + "_deserialize", DocUtils::ToDeclareDoc("const auto&", "meta_attrs"));
  for (const auto& a : plugin->attrs) {
    stack_.call_arg(a->name);
  }
  stack_.comment("extract meta inputs").declare("std::vector<MetaTensor>", "input_metas");
  for (const auto& t : plugin->inputs) {
    stack_.func_call("TVMUtils::ToMetaTensor", DocUtils::ToDeclareDoc("MetaTensor", "m_" + t->name))
        .call_arg(t->name)
        .func_call("push_back", "", "input_metas")
        .call_arg("m_input");
  }
  stack_.declare("std::vector<MetaTensor>", "output_metas");
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas");
  stack_.declare("Array<TensorStructInfo>", "output_sinfo");
  for (size_t i = 0; i < plugin->outputs.size(); i++) {
    const auto& struct_name = "s_" + plugin->outputs[i]->name;
    stack_.func_call("TVMUtils::ToTensorStructInfo", DocUtils::ToDeclareDoc("auto", struct_name))
        .call_arg(DocUtils::ToIndexDoc("output_metas", i));
    int device_idx = plugin->FindDeviceRefIdx(plugin->outputs[i]);
    if (device_idx >= 0) {
      stack_.call_arg(plugin->inputs[device_idx]->name);
    } else {
      stack_.call_arg("TVMUtils::ToTVMDevice(\"" + plugin->outputs[i]->device + "\")");
    }
    stack_.func_call("push_back", "", "output_sinfo").call_arg(struct_name);
  }
  stack_.func_end("output_sinfo");
  // infer layout
  stack_.func_def("InferLayout" + plugin->name, "InferLayoutOutput")
      .func_arg("inputs", "const Array<Expr>&")
      .func_arg("var_layout_map", "const VarLayoutMap&")
      .func_start()
      .comment("define attrs");
  for (size_t i = 0; i < plugin->attrs.size(); i++) {
    stack_.func_call("Downcast<StringImm>", "str_" + plugin->attrs[i]->name)
        .call_arg(DocUtils::ToIndexDoc("inputs", i + plugin->inputs.size()));
  }
  stack_.declare("Array<NLayout>", "input_layouts")
      .declare("Array<NLayout>", "output_layouts")
      .comment("extract meta attrs")
      .func_call(plugin->name + "_deserialize", "const " + attr_name + "& meta_attrs");
  for (const auto& a : plugin->attrs) {
    stack_.call_arg("str_" + a->name);
  }
  stack_.comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("LayoutUtils::InferLayoutDecision",
                 DocUtils::ToDeclareDoc("const auto&", "in_layout"))
      .call_arg(DocUtils::ToIndexDoc("inputs", "i"))
      .call_arg("var_layout_map")
      .func_call("TVMUtils::ToMetaTensor", DocUtils::ToDeclareDoc("MetaTensor", "m_input"))
      .call_arg(DocUtils::ToIndexDoc("inputs", "i"))
      .call_arg("in_layout")
      .func_call("push_back", "", "input_layouts")
      .call_arg("in_layout")
      .func_call("push_back", "", "input_metas")
      .call_arg("m_input")
      .for_end();
  stack_.declare("std::vector<MetaTensor>", "output_metas");
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas");
  stack_.for_start("i", 0, plugin->outputs.size())
      .func_call("LayoutDecision", DocUtils::ToDeclareDoc("const auto&", "out_layout"))
      .call_arg(
          DocUtils::ToAttrAccessDoc(DocUtils::ToIndexDoc("output_metas", "i"), "layout_name()"))
      .func_call("push_back", "", "output_layouts")
      .call_arg("out_layout")
      .for_end()
      .func_call("Attrs", DocUtils::ToDeclareDoc("auto", "new_attrs"))
      .call_arg(DocUtils::ToAttrAccessDoc(DocUtils::ToPtrDoc("call"), "attrs"))
      .func_call("InferLayoutOutput", DocUtils::ToDeclareDoc("const auto&", "infer_output"))
      .call_arg("input_layouts")
      .call_arg("output_layouts")
      .call_arg("new_attrs");
  stack_.func_end("infer_output");
  /*
  //  op make
  stack_.func_def(plugin->name, "Expr");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "Expr");
  }
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, ToTVMType(a->type));
  }
  stack_.func_start().func_call("make_object<" + r_attr_name + ">", "auto attrs");
  for (const auto& a : plugin->attrs) {
    stack_.assign("attrs->" + a->name, a->name);
  }
  stack_.func_call("Op::Get", "const Op& op")
      .call_arg(DocUtils::ToStrDoc(plugin->name))
      .declare("Array<Expr>", "in_args", 0, false);
  for (const auto& t : plugin->inputs) {
    stack_.declare_arg(t->name);
  }
  stack_.func_call("Call", DocUtils::ToDeclareDoc("const auto&", "call"))
      .call_arg("op")
      .call_arg("in_args")
      .call_arg("Attrs(attrs)")
      .call_arg("{}")
      .func_end("call");
  // register op make
  stack_.func_call("TVM_REGISTER_GLOBAL")
      .call_arg(DocUtils::ToStrDoc("msc.plugin.op." + plugin->name))
      .method_call("set_body_typed")
      .call_arg(plugin->name)
      .line();
  Array<String> infer_args{"input_metas", "meta_attrs", "false"};
  // infer struct info
  const auto ptr_args = DocUtils::ToAttrAccessDoc(DocUtils::ToPtrDoc("call"), "args");
  stack_.func_def("InferStructInfo" + plugin->name, "StructInfo")
      .func_arg("call", "const Call&")
      .func_arg("ctx", "const BlockBuilder&")
      .func_start()
      .comment("extract meta attrs")
      .func_call(plugin->name + "_attrs_to_meta",
                 DocUtils::ToDeclareDoc("const auto&", "meta_attrs"))
      .call_arg("call->attrs.as<" + r_attr_name + ">()")
      .comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("TVMUtils::ToMetaTensor", DocUtils::ToDeclareDoc("MetaTensor", "m_input"))
      .call_arg(DocUtils::ToIndexDoc(ptr_args, "i"))
      .func_call("push_back", "", "input_metas")
      .call_arg("m_input")
      .for_end();
  stack_.declare("std::vector<MetaTensor>", "output_metas");
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas");
  stack_.declare("Array<StructInfo>", "output_sinfo");
  for (size_t i = 0; i < plugin->outputs.size(); i++) {
    const auto& struct_name = "s_" + plugin->outputs[i]->name;
    stack_.func_call("TVMUtils::ToTensorStructInfo", DocUtils::ToDeclareDoc("auto", struct_name))
        .call_arg(DocUtils::ToIndexDoc("output_metas", i));
    int device_idx = plugin->FindDeviceRefIdx(plugin->outputs[i]);
    if (device_idx >= 0) {
      stack_.call_arg(DocUtils::ToIndexDoc(ptr_args, device_idx));
    } else {
      stack_.call_arg("TVMUtils::ToTVMDevice(\"" + plugin->outputs[i]->device + "\")");
    }
    stack_.func_call("push_back", "", "output_sinfo").call_arg(struct_name);
  }
  if (plugin->outputs.size() == 1) {
    stack_.func_end("output_sinfo[0]");
  } else {
    stack_.func_call("TupleStructInfo", DocUtils::ToDeclareDoc("const auto&", "tuple_info"))
        .call_arg("output_sinfo");
    stack_.func_end("tuple_info");
  }
  // infer layout
  stack_.func_def("InferLayout" + plugin->name, "InferLayoutOutput")
      .func_arg("call", "const Call&")
      .func_arg("desired_layouts", "const Map<String, Array<String>>&")
      .func_arg("var_layout_map", "const VarLayoutMap&")
      .func_start()
      .declare("Array<NLayout>", "input_layouts")
      .declare("Array<NLayout>", "output_layouts")
      .comment("extract meta attrs")
      .func_call(plugin->name + "_attrs_to_meta", "const " + attr_name + "& meta_attrs")
      .call_arg("call->attrs.as<" + r_attr_name + ">()")
      .comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("LayoutUtils::InferLayoutDecision",
                 DocUtils::ToDeclareDoc("const auto&", "in_layout"))
      .call_arg(DocUtils::ToIndexDoc(ptr_args, "i"))
      .call_arg("var_layout_map")
      .func_call("push_back", "", "input_layouts")
      .call_arg("in_layout")
      .func_call("TVMUtils::ToMetaTensor", DocUtils::ToDeclareDoc("MetaTensor", "m_input"))
      .call_arg(DocUtils::ToIndexDoc(ptr_args, "i"))
      .call_arg("in_layout")
      .func_call("push_back", "", "input_metas")
      .call_arg("m_input")
      .for_end();
  stack_.declare("std::vector<MetaTensor>", "output_metas");
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas");
  stack_.for_start("i", 0, plugin->outputs.size())
      .func_call("LayoutDecision", DocUtils::ToDeclareDoc("const auto&", "out_layout"))
      .call_arg(
          DocUtils::ToAttrAccessDoc(DocUtils::ToIndexDoc("output_metas", "i"), "layout_name()"))
      .func_call("push_back", "", "output_layouts")
      .call_arg("out_layout")
      .for_end()
      .func_call("Attrs", DocUtils::ToDeclareDoc("auto", "new_attrs"))
      .call_arg(DocUtils::ToAttrAccessDoc(DocUtils::ToPtrDoc("call"), "attrs"))
      .func_call("InferLayoutOutput", DocUtils::ToDeclareDoc("const auto&", "infer_output"))
      .call_arg("input_layouts")
      .call_arg("output_layouts")
      .call_arg("new_attrs");
  stack_.func_end("infer_output");
  */
}

void TVMPluginCodeGen::CodeGenOpRegister(const Plugin& plugin) {
  stack_.func_call("TVM_REGISTER_GLOBAL")
      .call_arg(DocUtils::ToStrDoc("msc.plugin.op.InferStructInfo" + plugin->name))
      .method_call("set_body_typed")
      .call_arg("InferStructInfo" + plugin->name)
      .line()
      .func_call("TVM_REGISTER_GLOBAL")
      .call_arg(DocUtils::ToStrDoc("msc.plugin.op.InferLayout" + plugin->name))
      .method_call("set_body_typed")
      .call_arg("InferLayout" + plugin->name)
      .line();

  // const auto& r_attr_name = RelaxMetaAttrCls(plugin);
  /*
  stack_.func_call("TVM_REGISTER_OP")
      .call_arg(DocUtils::ToStrDoc(plugin->name))
      .method_call("set_num_inputs", true)
      .call_arg(plugin->inputs.size());
  for (const auto& t : plugin->inputs) {
    stack_.method_call("add_argument", true)
        .call_arg(DocUtils::ToStrDoc(t->name))
        .call_arg(DocUtils::ToStrDoc("Tensor"))
        .call_arg(DocUtils::ToStrDoc(t->describe));
  }
  stack_.method_call("set_attrs_type<" + r_attr_name + ">", true)
      .method_call("set_attr<FInferStructInfo>", true)
      .call_arg(DocUtils::ToStrDoc("FInferStructInfo"))
      .call_arg("InferStructInfo" + plugin->name)
      .method_call("set_attr<FRelaxInferLayout>", true)
      .call_arg(DocUtils::ToStrDoc("FRelaxInferLayout"))
      .call_arg("InferLayout" + plugin->name)
      .method_call("set_attr<Bool>", true)
      .call_arg(DocUtils::ToStrDoc("FPurity"))
      .call_arg("Bool(true)")
      .line();
  */
}

void TVMPluginCodeGen::CodeGenOpRuntime(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  const auto& func_name =ComputeName(plugin);
  String device_cond = "";
  for (size_t i = 0; i < plugin->inputs.size(); i++) {
    if (plugin->inputs[i]->device == "cuda" || plugin->inputs[i]->device == "default") {
      device_cond = device_cond + "TMVUtils::OnDevice(" + plugin->inputs[i]->name +
                    ", DLDeviceType::kDLCUDA)";
    } else {
      device_cond = device_cond + device_cond + "TMVUtils::OnDevice(" + plugin->inputs[i]->name +
                    ", DLDeviceType::kDLCPU)";
    }
    device_cond = device_cond + (i == plugin->inputs.size() - 1 ? "" : " && ");
  }
  stack_.func_def("_" + func_name);
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "DLTensor*");
  }
  for (const auto& t : plugin->outputs) {
    stack_.func_arg(t->name, "DLTensor*");
  }
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, "const StringImm&");
  }
  stack_.func_start()
      .comment("extract meta attrs")
      .func_call(plugin->name + "_deserialize", "const " + attr_name + "& meta_attrs");
  for (const auto& a : plugin->attrs) {
    stack_.call_arg(a->name);
  }
  stack_.comment("do the compute").cond_if(device_cond);
  CodeGenCompute(plugin, "cuda");
  stack_.cond_else();
  CodeGenCompute(plugin, "cpu");
  stack_.cond_end().func_end().line();
  // register the compute
  stack_.func_call("TVM_DLL_EXPORT_TYPED_FUNC")
      .call_arg(func_name)
      .call_arg("_" + func_name)
      .line();
}

void TVMPluginCodeGen::CodeGenCmake(const std::set<String>& devices) {
  stack_.line("cmake_minimum_required(VERSION " + config()->cmake_version + " FATAL_ERROR)")
      .line("project(msc_tvm_plugin)");
  if (devices.count("cuda")) {
    stack_.line("find_package(CUDA)").line("add_definitions(-DPLUGIN_ENABLE_CUDA)");
  }
  stack_.line("set(CMAKE_CXX_STANDARD 17)");
  stack_.line("set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -Wno-macro-redefined\")");
  stack_.line("add_definitions(-DPLUGIN_SUPPORT_TVM)");
  for (const auto& pair : config()->flags) {
    if (pair.second == "") {
      stack_.line("add_definitions(-D" + pair.first + ")");
    } else {
      stack_.line("add_definitions(-D" + pair.first + "=" + pair.second + ")");
    }
  }
  stack_.line("file(GLOB_RECURSE TVM_CC_SRCS src/*.cc)");
  if (devices.count("cuda")) {
    stack_.line("file(GLOB_RECURSE TVM_CU_SRCS src/*.cu)");
  }
  if (devices.count("cuda")) {
    stack_.line("cuda_add_library(msc_tvm_plugin SHARED ${TVM_CC_SRCS} ${TVM_CU_SRCS})");
  } else {
    stack_.line("add_library(msc_tvm_plugin SHARED ${TVM_CC_SRCS})");
  }
  stack_.line("set(TVM_ROOT " + config()->tvm_root + ")");
  String includes = "${TVM_ROOT}/include";
  includes = includes + " ${TVM_ROOT}/3rdparty/dmlc-core/include";
  includes = includes + " ${TVM_ROOT}/3rdparty/dlpack/include";
  includes = includes + " ${TVM_ROOT}/3rdparty/compiler-rt";
  if (config()->includes.size() > 0) {
    for (const auto& include : config()->includes) {
      includes = includes + " " + include;
    }
  }
  stack_.line("target_include_directories(msc_tvm_plugin PUBLIC " + includes + ")")
      .line("find_library(TVM_LIB NAMES tvm HINTS ${TVM_ROOT}/build NO_DEFAULT_PATH)");
  String libs = "";
  for (const auto& lib : config()->libs) {
    libs = libs + " " + lib;
  }
  stack_.line("target_link_libraries(msc_tvm_plugin ${TVM_LIB}" + libs + ")");
  if (config()->install_dir.size() > 0) {
    stack_.line("SET(LIBRARY_OUTPUT_PATH " + config()->install_dir + ")");
    if (config()->libs.size() > 0) {
      stack_.line("file(COPY " + libs + " DESTINATION " + config()->install_dir + ")");
    }
  }
}

void TVMPluginCodeGen::CodeGenManagerImports() {
  stack_.line("import tvm")
      .line("from tvm import relax")
      .line("from tvm.relax import call_dps_packed")
      .line("from tvm.tir import expr as tir_expr")
      .line("from tvm.contrib.msc.core import utils as msc_utils");
  BasePluginCodeGen<TVMPluginCodeGenConfig>::CodeGenManagerImports();
}

void TVMPluginCodeGen::CodeGenManagerMethods() {
  stack_.func_def("__init__")
      .func_arg("self", "object")
      .func_arg("lib_folder", "str", "None")
      .func_start()
      .cond_if("lib_folder is None")
      .assign("root", "os.path.dirname(__file__)")
      .assign(DocUtils::ToAttrAccessDoc("self", "_lib_folder"), "os.path.join(root, \"libs\")")
      .cond_else()
      .assign(DocUtils::ToAttrAccessDoc("self", "_lib_folder"), "lib_folder")
      .cond_end()
      .line("assert os.path.isdir(self._lib_folder), \"lib_folder not exist\"")
      .for_start("lib", "os.listdir(self._lib_folder)")
      .assign("lib_file", "os.path.join(self._lib_folder, lib)")
      .func_call("CDLL", "", "ctypes")
      .call_arg("lib_file")
      .for_end()
      .line("from tvm.contrib.msc.plugin.op import _ffi_api")
      .assign(DocUtils::ToAttrAccessDoc("self", "_ffi_api"), "_ffi_api")
      .func_end();
  BasePluginCodeGen<TVMPluginCodeGenConfig>::CodeGenManagerMethods();
}

void TVMPluginCodeGen::CodeGenPluginManager(const Plugin& plugin) {
  stack_.func_def(plugin->name).func_arg("self", "object");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "relax.Expr");
  }
  for (const auto& attr : plugin->attrs) {
    stack_.func_arg(attr->name, ToPyType(attr->type), attr->default_value);
  }
  stack_.func_arg("name", "str", "\"" + plugin->name + "\"").func_start();
  for (const auto& a : plugin->attrs) {
    stack_.func_call("to_string", a->name)
        .call_arg(a->name)
        .func_call("relax.StringImm", a->name)
        .call_arg(a->name);
  }
  /*
  ret_tensor = _wrap_nested(
                        call_dps_packed(
                            func_name,
                            args=RxTuple(relax_args),
                            out_sinfo=out_sinfo,
                        ),
                        func_name,
                    )
                    */
  /*
  for (const auto& attr : plugin->attrs) {
    const String& type_cls = "tir_expr." + ToTVMType(attr->type);
    stack_.cond_if("not isinstance(" + attr->name + ", " + type_cls + ")")
        .func_call(type_cls, attr->name);
    if (attr->type == "float" || attr->type == "float32") {
      stack_.call_arg(DocUtils::ToStrDoc("float32"));
    } else if (attr->type == "double" || attr->type == "float64") {
      stack_.call_arg(DocUtils::ToStrDoc("float64"));
    }
    stack_.call_arg(attr->name).cond_end();
  }
  stack_.func_call(plugin->name, "op", "self._ffi_api");
  for (const auto& t : plugin->inputs) {
    stack_.call_arg(t->name);
  }
  for (const auto& attr : plugin->attrs) {
    stack_.call_arg(attr->name);
  }
  stack_.func_call("msc_utils.set_expr_name", "op")
      .call_arg("op")
      .call_arg("name")
      .func_end("op")
      .comment(GetPyComment(plugin), true);
  */
}

void TVMPluginCodeGen::CodeGenCompute(const Plugin& plugin, const String& device) {
  if (plugin->externs.count(device + "_compute")) {
    // compute with dtype
    auto prepare_tensor = [this, &device](const PluginTensor& tensor,
                                          const Map<String, String>& dtypes, size_t idx,
                                          const String& collect) {
      const String& t_name = "d_" + tensor->name;
      const String& t_dtype = dtypes.count(tensor->name) ? dtypes[tensor->name] : tensor->dtype;
      const String& tensor_type = "DataTensor<" + t_dtype + ">";
      const String& anno = collect == "input" ? "const " + tensor_type + "&" : tensor_type;
      stack_.func_call("TVMUtils::To" + tensor_type, DocUtils::ToDeclareDoc(anno, t_name))
          .call_arg(tensor->name)
          .call_arg(collect == "input" ? "true" : "false");
      return t_name;
    };
    for (const auto& dtypes : plugin->GetDtypeMatrix()) {
      Array<String> compute_args;
      const auto& tensor_dtypes = GetTensorDtypes(plugin, dtypes);
      String dtype_cond = "";
      size_t cnt = 0;
      for (const auto& pair : dtypes) {
        const auto& t_name = plugin->inputs[pair.first->value]->name;
        dtype_cond = dtype_cond + "TVMUtils::ToMetaType(" + t_name +
                     "->dtype) == DataUtils::ToMetaType(\"" + pair.second + "\")";
        dtype_cond = dtype_cond + (cnt == dtypes.size() - 1 ? "" : " && ");
        cnt++;
      }
      // prepare compute datas
      stack_.cond_if(dtype_cond).comment("prepare compute datas");
      for (size_t i = 0; i < plugin->inputs.size(); i++) {
        const String& t_name = prepare_tensor(plugin->inputs[i], tensor_dtypes, i, "input");
        compute_args.push_back(t_name);
      }
      for (size_t i = 0; i < plugin->outputs.size(); i++) {
        const String& t_name = prepare_tensor(plugin->outputs[i], tensor_dtypes, i, "output");
        compute_args.push_back(t_name);
      }
      ICHECK(plugin->buffers.size() == 0) << "Plugin with buffers is not supported in tvm";
      compute_args.push_back("meta_attrs");
      if (device == "cuda") {
        stack_.assign("stream", "runtime::CUDAThreadEntry::ThreadLocal()->stream", "auto");
        compute_args.push_back("stream");
      }
      CodeGenSafeCall(plugin->externs[device + "_compute"], compute_args);
      stack_.cond_end();
    }
    /*
    auto prepare_tensor = [this, &device](const PluginTensor& tensor,
                                          const Map<String, String>& dtypes, size_t idx,
                                          const String& collect) {
      const String& t_name = "d_" + tensor->name;
      const String& t_dtype = dtypes.count(tensor->name) ? dtypes[tensor->name] : tensor->dtype;
      stack_.func_call("TVMUtils::CheckDevice").call_arg(tensor->name);
      if (tensor->device == "cuda" || (tensor->device == "default" && device == "cuda")) {
        stack_.call_arg("DLDeviceType::kDLCUDA");
      } else if (tensor->device == "cpu" || (tensor->device == "default" && device == "cpu")) {
        stack_.call_arg("DLDeviceType::kDLCPU");
      }
      const String& anno = collect == "input" ? "const auto&" : "auto";
      stack_
          .func_call("TVMUtils::ToDataTensor<" + t_dtype + ">",
                     DocUtils::ToDeclareDoc(anno, t_name))
          .call_arg(tensor->name)
          .call_arg(collect == "input" ? "true" : "false");
      return t_name;
    };
    const auto& compute_func = plugin->name + "_" + device + "_compute";
    const auto& attr_name = MetaAttrCls(plugin);
    stack_.comment(device + " compute function").func_def(compute_func);
    for (const auto& t : plugin->inputs) {
      stack_.func_arg(t->name, "DLTensor*");
    }
    for (const auto& t : plugin->outputs) {
      stack_.func_arg(t->name, "DLTensor*");
    }
    for (const auto& a : plugin->attrs) {
      stack_.func_arg(a->name, ToTVMType(a->type));
    }
    stack_.func_start()
        .comment("extract meta attrs")
        .func_call(plugin->name + "_args_to_meta", "const " + attr_name + "& meta_attrs");
    for (const auto& a : plugin->attrs) {
      stack_.call_arg(a->name);
    }
    // compute with dtype
    for (const auto& dtypes : plugin->GetDtypeMatrix()) {
      Array<String> compute_args;
      const auto& tensor_dtypes = GetTensorDtypes(plugin, dtypes);
      String dtype_cond = "";
      size_t cnt = 0;
      for (const auto& pair : dtypes) {
        const auto& t_name = plugin->inputs[pair.first->value]->name;
        dtype_cond = dtype_cond + "TVMUtils::ToMetaType(" + t_name +
                     "->dtype) == DataUtils::ToMetaType(\"" + pair.second + "\")";
        dtype_cond = dtype_cond + (cnt == dtypes.size() - 1 ? "" : " && ");
        cnt++;
      }
      // prepare compute datas
      stack_.cond_if(dtype_cond).comment("prepare compute datas");
      for (size_t i = 0; i < plugin->inputs.size(); i++) {
        const String& t_name = prepare_tensor(plugin->inputs[i], tensor_dtypes, i, "input");
        compute_args.push_back(t_name);
      }
      for (size_t i = 0; i < plugin->outputs.size(); i++) {
        const String& t_name = prepare_tensor(plugin->outputs[i], tensor_dtypes, i, "output");
        compute_args.push_back(t_name);
      }
      ICHECK(plugin->buffers.size() == 0) << "Plugin with buffers is not supported in tvm";
      compute_args.push_back("meta_attrs");
      if (device == "cuda") {
        stack_.assign("stream", "runtime::CUDAThreadEntry::ThreadLocal()->stream", "auto");
        compute_args.push_back("stream");
      }
      CodeGenSafeCall(plugin->externs[device + "_compute"], compute_args);
      stack_.cond_end();
    }
    stack_.func_end();
    // register runtime
    stack_.func_call("TVM_REGISTER_GLOBAL")
        .call_arg(DocUtils::ToStrDoc("msc.plugin." + plugin->name + "_" + device + "_compute"))
        .method_call("set_body_typed")
        .call_arg(compute_func)
        .line();
    */
  }
}

TVM_REGISTER_GLOBAL("msc.plugin.GetTVMPluginSources")
    .set_body_typed([](const String& codegen_config, const String& print_config,
                       const String& codegen_type) -> Map<String, String> {
      TVMPluginCodeGen codegen = TVMPluginCodeGen(codegen_config);
      if (codegen_type == "build") {
        return codegen.GetBuildSources(print_config);
      }
      if (codegen_type == "manager") {
        return codegen.GetManagerSources(print_config);
      }
      return Map<String, String>();
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
