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
  const auto& attr_name = AttrClsName(plugin);
  const auto& r_attr_name = RelaxAttrClsName(plugin);
  stack_.line().struct_start(r_attr_name + " : tvm::AttrsNode<" + r_attr_name + ">");
  // define attributes
  stack_.comment("define attributes");
  for (const auto& a : plugin->attrs) {
    stack_.declare(ConvertTVMAttrType(a->type), a->name);
    if (a->default_value.size() > 0) {
      stack_.declare_arg(a->default_value);
    }
  }
  stack_.line()
      .comment("register attributes")
      .func_def("TVM_DECLARE_ATTRS", CppPrinter::Empty())
      .func_arg(r_attr_name, CppPrinter::Empty())
      .func_arg("\"relax.attrs." + r_attr_name + "\"", CppPrinter::Empty())
      .func_start();
  for (const auto& a : plugin->attrs) {
    stack_.func_call("TVM_ATTR_FIELD")
        .call_arg(a->name)
        .method_call("describe")
        .call_arg(DocUtils::ToStrDoc(a->describe));
  }
  stack_.func_end().struct_end();
  // convert method to meta
  stack_.line()
      .comment("convert to meta method")
      .func_def(r_attr_name + "_to_meta", attr_name)
      .func_arg("attrs", "const " + r_attr_name + "&");
  // convert method from meta
  stack_.comment("convert from meta method")
      .func_def(r_attr_name + "_from_meta", r_attr_name)
      .func_arg("meta_attrs", "const " + attr_name + "&");
}

void TVMPluginCodeGen::CodeGenAttrDefine(const Plugin& plugin) {
  const auto& attr_name = AttrClsName(plugin);
  const auto& r_attr_name = RelaxAttrClsName(plugin);
  // convert method to meta
  stack_.line()
      .comment("convert to meta method")
      .func_def(r_attr_name + "_to_meta", attr_name)
      .func_arg("attrs", "const " + r_attr_name + "&")
      .func_start()
      .func_call(attr_name, "meta_attrs");
  for (const auto& a : plugin->attrs) {
    if (IsListType(a->type)) {
      stack_.for_start("v", "attrs." + a->name)
          .func_call("push_back", "meta_attrs." + a->name)
          .call_arg("v->value")
          .for_end();
    } else if (a->type == "string") {
      stack_.func_call("std::string", "meta_attrs." + a->name).call_arg("attrs." + a->name);
    } else {
      stack_.assign("meta_attrs." + a->name, "attrs." + a->name);
    }
  }
  stack_.func_end("meta_attrs");
  // convert method from meta
  stack_.comment("convert from meta method")
      .func_def(r_attr_name + "_from_meta", r_attr_name)
      .func_arg("meta_attrs", "const " + attr_name + "&")
      .func_start()
      .func_call("make_object<" + r_attr_name + ">", "auto attrs");
  for (const auto& a : plugin->attrs) {
    if (IsListType(a->type)) {
      const auto& ele_type = GetEleType(a->type);
      stack_.for_start("v", "meta_attrs." + a->name).func_call("push_back", "attrs." + a->name);
      if (ele_type == "bool") {
        stack_.call_arg("Bool(v)");
      } else if (ele_type == "int32" || ele_type == "int") {
        stack_.call_arg("Integer(v)");
      } else {
        stack_.call_arg("FloatImm(v)");
      }
      stack_.for_end();
    } else {
      stack_.assign("attrs." + a->name, "meta_attrs." + a->name);
    }
  }
  stack_.func_end("attrs");
}

void TVMPluginCodeGen::CodeGenOpDefine(const Plugin& plugin) {
  const auto& attr_name = AttrClsName(plugin);
  const auto& r_attr_name = RelaxAttrClsName(plugin);
  // register attrs
  stack_.func_call("TVM_REGISTER_NODE_TYPE").call_arg(RelaxAttrClsName(plugin)).line();
  // op make
  stack_.func_def(plugin->name, "Expr");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "Expr");
  }
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, ConvertTVMAttrType(a->type));
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
  stack_.func_call("Call", "const auto& call")
      .call_arg("op")
      .call_arg("in_args")
      .call_arg("Attrs(attrs)")
      .call_arg("{}")
      .func_end("call");
  // register op make
  stack_.func_call("TVM_REGISTER_GLOBAL")
      .call_arg(DocUtils::ToStrDoc("msc.plugin." + plugin->name))
      .method_call(plugin->name)
      .line();
  // infer func
  stack_.func_def("InferStructInfo" + plugin->name)
      .func_arg("call", "const Call&")
      .func_arg("ctx", "const BlockBuilder&")
      .func_start()
      .func_call(r_attr_name + "_to_meta", "const " + attr_name + "& meta_attrs")
      .call_arg("Downcast<" + r_attr_name + ">(call->attrs)")
      .line()
      .comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("TVMUtils::ToMetaTensor", "MetaTensor m_input")
      .call_arg(DocUtils::ToIndexDoc("call->args", "i"))
      .func_call("push_back", "", "input_metas")
      .call_arg("m_input")
      .for_end();
  Array<String> call_args{"input_metas", "meta_attrs", "false"};
  CodeGenSafeCall(plugin->externs["infer_output"], call_args, "output_metas");
  stack_.declare("Array<StructInfo>", "output_sinfo")
      .for_start("o", "output_metas")
      .func_call("TVMUtils::ToTensorStructInfo", "o_info")
      .call_arg("o")
      .func_call("push_back", "output_sinfo")
      .call_arg("o_info")
      .for_end();
  if (plugin->outputs.size() == 1) {
    stack_.func_end("output_sinfo[0]");
  } else {
    stack_.func_call("TupleStructInfo", "const auto& tuple_info").call_arg("output_sinfo");
    stack_.func_end("tuple_info");
  }
}

void TVMPluginCodeGen::CodeGenOpRegister(const Plugin& plugin) {}

void TVMPluginCodeGen::CodeGenCmake(const std::set<String>& devices) {
  stack_.line("cmake_minimum_required(VERSION " + config()->cmake_version + " FATAL_ERROR)")
      .line("project(msc_tvm_plugin)");
  if (devices.count("cuda")) {
    stack_.line("find_package(CUDA)").line("add_definitions(-DPLUGIN_ENABLE_CUDA)");
  }
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
    stack_.line("cuda_add_library(msc_tvm_plugin SHARED ${TORCH_CC_SRCS} ${TORCH_CU_SRCS})");
  } else {
    stack_.line("add_library(msc_tvm_plugin SHARED ${TORCH_CC_SRCS})");
  }
  if (config()->includes.size() > 0) {
    String includes = "";
    for (const auto& include : config()->includes) {
      includes = includes + " " + include;
    }
    stack_.line("target_include_directories(msc_tvm_plugin PUBLIC" + includes + ")");
  }
  String libs = "";
  for (const auto& lib : config()->libs) {
    libs = libs + " " + lib;
  }
  stack_.line("target_link_libraries(msc_tvm_plugin " + libs + ")");
  if (config()->install_dir.size() > 0) {
    stack_.line("SET(LIBRARY_OUTPUT_PATH " + config()->install_dir + ")");
    if (libs.size() > 0) {
      stack_.line("file(COPY " + libs + " DESTINATION " + config()->install_dir + ")");
    }
  }
}

void TVMPluginCodeGen::CodeGenManagerImports() {
  stack_.line("import tvm");
  BasePluginCodeGen<TVMPluginCodeGenConfig>::CodeGenManagerImports();
}

void TVMPluginCodeGen::CodeGenManagerMethods() {
  stack_.func_def("__init__")
      .func_arg("self", "object")
      .func_arg("lib_folder", "str", "None")
      .func_start()
      .cond_if("lib_folder is None")
      .assign("root", "os.path.dirname(__file__)")
      .assign("self._lib_folder", "os.path.join(root, \"libs\")")
      .cond_else()
      .assign("self._lib_folder", "lib_folder")
      .cond_end()
      .line("assert os.path.isdir(self._lib_folder), \"lib_folder not exist\"")
      .for_start("lib", "os.listdir(self._lib_folder)")
      .assign("lib_file", "os.path.join(self._lib_folder, lib)")
      .func_call("CDLL", "", "ctypes")
      .call_arg("lib_file")
      .for_end()
      .func_end();
  BasePluginCodeGen<TVMPluginCodeGenConfig>::CodeGenManagerMethods();
}

void TVMPluginCodeGen::CodeGenPluginManager(const Plugin& plugin) {}

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
