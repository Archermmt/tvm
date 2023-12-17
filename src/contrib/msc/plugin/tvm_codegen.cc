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
  stack_.struct_start(r_attr_name + " : tvm::AttrsNode<" + r_attr_name + ">");
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
      .func_def(r_attr_name + "_to_meta", attr_name)
      .func_arg("attrs", "const " + r_attr_name + "&");
  // convert method to meta in runtime
  stack_.comment("convert arguments to meta method")
      .func_def(plugin->name + "_args_to_meta", attr_name)
      .func_arg("args", "TVMArgs")
      .func_arg("start", "size_t");
  // convert method from meta
  stack_.comment("convert relax attrs from meta method")
      .func_def(r_attr_name + "_from_meta", r_attr_name)
      .func_arg("meta_attrs", "const " + attr_name + "&");
}

void TVMPluginCodeGen::CodeGenAttrDefine(const Plugin& plugin) {
  const auto& attr_name = AttrClsName(plugin);
  const auto& r_attr_name = RelaxAttrClsName(plugin);
  // convert method to meta
  stack_.func_def(r_attr_name + "_to_meta", attr_name)
      .func_arg("attrs", "const " + r_attr_name + "&")
      .func_start()
      .declare(attr_name, "meta_attrs");
  for (const auto& a : plugin->attrs) {
    if (IsListType(a->type)) {
      stack_.for_start("v", "attrs." + a->name)
          .func_call("push_back", NullOpt, DocUtils::ToAttrAccessDoc("meta_attrs", a->name))
          .call_arg("v->value")
          .for_end();
    } else if (a->type == "string") {
      stack_.func_call("std::string", "meta_attrs." + a->name)
          .call_arg(DocUtils::ToAttrAccessDoc("attrs", a->name));
    } else {
      stack_.assign(DocUtils::ToAttrAccessDoc("meta_attrs", a->name),
                    DocUtils::ToAttrAccessDoc("attrs", a->name));
    }
  }
  stack_.func_end("meta_attrs");
  // convert method to meta in runtime
  stack_.func_def(plugin->name + "_args_to_meta", attr_name)
      .func_arg("args", "TVMArgs")
      .func_arg("pos", "size_t")
      .func_start()
      .declare(attr_name, "meta_attrs");
  for (const auto& a : plugin->attrs) {
    if (IsListType(a->type)) {
      const auto& ele_type = GetEleType(a->type);
      stack_.assign(a->name + "_size", DocUtils::ToIndexDoc("args", "pos"), "size_t")
          .line("pos++;");
      stack_.for_start("i", "0", a->name + "_size")
          .func_call(FromTVMArgType(ele_type), "const auto& arg")
          .func_call("push_back", NullOpt, DocUtils::ToAttrAccessDoc("meta_attrs", a->name))
          .call_arg("arg")
          .for_end()
          .assign("pos", "pos + " + a->name + "_size");
    } else {
      stack_.func_call(FromTVMArgType(a->type), "meta_attrs." + a->name)
          .call_arg(DocUtils::ToIndexDoc("args", "pos"))
          .line("pos++;");
    }
  }
  stack_.func_end("meta_attrs");
  // convert method from meta
  stack_.func_def(r_attr_name + "_from_meta", r_attr_name)
      .func_arg("meta_attrs", "const " + attr_name + "&")
      .func_start()
      .func_call("make_object<" + r_attr_name + ">", "auto attrs");
  for (const auto& a : plugin->attrs) {
    if (IsListType(a->type)) {
      const auto& ele_type = GetEleType(a->type);
      stack_.for_start("v", "meta_attrs." + a->name).func_call("push_back", "attrs->" + a->name);
      if (ele_type == "bool") {
        stack_.call_arg("Bool(v)");
      } else if (ele_type == "int32" || ele_type == "int") {
        stack_.call_arg("Integer(v)");
      } else {
        stack_.call_arg("FloatImm(v)");
      }
      stack_.for_end();
    } else {
      stack_.assign("attrs->" + a->name, DocUtils::ToAttrAccessDoc("meta_attrs", a->name));
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
  stack_.func_call("Call", "const auto& call")
      .call_arg("op")
      .call_arg("in_args")
      .call_arg("Attrs(attrs)")
      .call_arg("{}")
      .func_end("call");
  // register op make
  stack_.func_call("TVM_REGISTER_GLOBAL")
      .call_arg(DocUtils::ToStrDoc("msc.plugin." + plugin->name))
      .method_call("set_body_typed")
      .call_arg(plugin->name)
      .line();
  Array<String> infer_args{"input_metas", "meta_attrs", "false"};
  // infer struct info
  stack_.func_def("InferStructInfo" + plugin->name)
      .func_arg("call", "const Call&")
      .func_arg("ctx", "const BlockBuilder&")
      .func_start()
      .comment("extract meta attrs")
      .func_call(r_attr_name + "_to_meta", "const " + attr_name + "& meta_attrs")
      .call_arg("Downcast<" + r_attr_name + ">(call->attrs)")
      .comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("TVMUtils::ToMetaTensor", "MetaTensor m_input")
      .call_arg(DocUtils::ToIndexDoc("call->args", "i"))
      .func_call("push_back", "", "input_metas")
      .call_arg("m_input")
      .for_end();
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas");
  stack_.declare("Array<StructInfo>", "output_sinfo");
  for (size_t i = 0; i < plugin->outputs.size(); i++) {
    const auto& struct_name = "s_" + plugin->outputs[i]->name;
    stack_.func_call("TorchUtils::ToTensorStructInfo", "auto " + struct_name)
        .call_arg(DocUtils::ToIndexDoc("output_metas", i));
    int device_idx = plugin->FindDeviceRefIdx(plugin->outputs[i]);
    if (device_idx >= 0) {
      const auto& input_doc = DocUtils::ToIndexDoc("input_tensors", device_idx);
      stack_.call_arg(DocUtils::ToAttrAccessDoc(input_doc, "device"));
    } else {
      stack_.call_arg("TorchUtils::ToTVMDevice(\"" + plugin->outputs[i]->device + "\")");
    }
    stack_.func_call("push_back", "", "output_sinfo").call_arg(struct_name);
  }
  if (plugin->outputs.size() == 1) {
    stack_.func_end("output_sinfo[0]");
  } else {
    stack_.func_call("TupleStructInfo", "const auto& tuple_info").call_arg("output_sinfo");
    stack_.func_end("tuple_info");
  }
  // infer layout
  stack_.func_def("InferLayout" + plugin->name)
      .func_arg("call", "const Call&")
      .func_arg("desired_layouts", "const Map<String, Array<String>>&")
      .func_arg("var_layout_map", "const VarLayoutMap&")
      .func_start()
      .declare("Array<NLayout>", "input_layouts")
      .declare("Array<NLayout>", "output_layouts")
      .comment("extract meta attrs")
      .func_call(r_attr_name + "_to_meta", "const " + attr_name + "& meta_attrs")
      .call_arg("Downcast<" + r_attr_name + ">(call->attrs)")
      .comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("LayoutUtils::InferLayoutDecision", "const auto& in_layout")
      .call_arg(DocUtils::ToIndexDoc("call->args", "i"))
      .call_arg("var_layout_map")
      .func_call("push_back", "", "input_layouts")
      .call_arg("in_layout")
      .func_call("TVMUtils::ToMetaTensor", "MetaTensor m_input")
      .call_arg(DocUtils::ToIndexDoc("call->args", "i"))
      .call_arg("in_layout")
      .func_call("push_back", "", "input_metas")
      .call_arg("m_input")
      .for_end();
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas");
  stack_.for_start("i", 0, plugin->outputs.size())
      .func_call("LayoutDecision", "const auto& out_layout")
      .call_arg(DocUtils::ToAttrAccessDoc(DocUtils::ToIndexDoc("output_metas", "i"), "layout()"))
      .func_call("push_back", "", "output_layouts")
      .call_arg("out_layout")
      .for_end()
      .func_call("Attrs", "new_attrs")
      .call_arg("call->args")
      .func_call("InferLayoutOutput", "const auto& infer_output")
      .call_arg("input_layouts")
      .call_arg("output_layouts")
      .call_arg("new_attrs");
  stack_.func_end("infer_output");
}

void TVMPluginCodeGen::CodeGenOpRegister(const Plugin& plugin) {
  const auto& r_attr_name = RelaxAttrClsName(plugin);
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
      .call_arg("Bool(true)");
}

void TVMPluginCodeGen::CodeGenOpRuntime(const Plugin& plugin) {
  CodeGenCompute(plugin, "cpu");
  CodeGenCompute(plugin, "cuda");
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
      .line("link_directories(${TVM_ROOT}/build)");
  String libs = " ";
  for (const auto& lib : config()->libs) {
    libs = libs + " " + lib;
  }
  stack_.line("target_link_libraries(msc_tvm_plugin" + libs + ")");
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

void TVMPluginCodeGen::CodeGenCompute(const Plugin& plugin, const String& device) {
  if (plugin->externs.count(device + "_compute")) {
    auto prepare_tensor = [this, &device](const PluginTensor& tensor,
                                          const Map<String, String>& dtypes, size_t idx,
                                          const String& collect) {
      const String& t_name = "d_" + tensor->name;
      const String& t_dtype = dtypes.count(tensor->name) ? dtypes[tensor->name] : tensor->dtype;
      const auto& arg_doc = DocUtils::ToIndexDoc("args", idx);
      stack_.func_call("TVMUtils::CheckDevice").call_arg(arg_doc);
      if (tensor->device == "cuda" || (tensor->device == "default" && device == "cuda")) {
        stack_.call_arg("DLDeviceType::KDLCUDA");
      } else if (tensor->device == "cpu" || (tensor->device == "default" && device == "cpu")) {
        stack_.call_arg("DLDeviceType::KDLCPU");
      }
      stack_.func_call("TVMUtils::ToDataTensor<" + t_dtype + ">", t_name)
          .call_arg(arg_doc)
          .call_arg(collect == "input" ? "true" : "false");
      return t_name;
    };
    const auto& compute_func = plugin->name + "_" + device + "_compute";
    const auto& attr_name = AttrClsName(plugin);
    stack_.comment(device + " compute function")
        .func_def(compute_func)
        .func_arg("args", "TVmArgs")
        .func_arg("ret", "TVMRetValue*")
        .func_start()
        .comment("extract meta attrs")
        .func_call(plugin->name + "_args_to_meta", "const " + attr_name + "& meta_attrs")
        .call_arg("args")
        .call_arg(plugin->inputs.size() + plugin->outputs.size());
    // compute with dtype
    for (const auto& dtypes : plugin->GetDtypeMatrix()) {
      Array<String> compute_args;
      const auto& tensor_dtypes = GetTensorDtypes(plugin, dtypes);
      String dtype_cond = "";
      size_t cnt = 0;
      for (const auto& pair : dtypes) {
        dtype_cond = dtype_cond + "TVMUtils::ToMetaType(args[" + std::to_string(pair.first->value) +
                     "]->dtype) == DataUtils::ToMetaType(\"" + pair.second + "\")";
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
        const String& t_name =
            prepare_tensor(plugin->outputs[i], tensor_dtypes, plugin->inputs.size() + i, "output");
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
