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
 * \file src/contrib/msc/plugin/torch_codegen.cc
 */
#include "torch_codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

void TorchPluginCodeGen::CodeGenAttrSerialize(const Plugin& plugin) {
  stack_.comment("serialize method")
      .func_def("Serialize", "std::vector<std::string>")
      .func_start()
      .declare("std::vector<std::string>", "attrs");
  for (const auto& attr : plugin->attrs) {
    stack_.func_call("push_back", "", "attrs")
        .call_arg("SerializeUtils::ToString(" + attr->name + ")");
  }
  stack_.func_end("attrs");
  stack_.comment("deserialize method")
      .func_def("Deserialize")
      .func_arg("attrs", "const std::vector<std::string>&")
      .func_start();
  for (size_t i = 0; i < plugin->attrs.size(); i++) {
    stack_.func_call("SerializeUtils::FromString")
        .call_arg(DocUtils::ToIndexDoc("attrs", i))
        .call_arg(plugin->attrs[i]->name);
  }
  stack_.func_end();
}

void TorchPluginCodeGen::CodeGenDefine(const Plugin& plugin) {
  stack_.struct_start(plugin->name + "_op : torch::CustomClassHolder");
  // define constructor
  stack_.constructor_def(plugin->name + "_op")
      .constructor_arg("attrs", "const std::vector<std::string>&")
      .constructor_start()
      .comment("get attributes")
      .func_call("Deserialize", "", "meta_attrs_")
      .call_arg("attrs")
      .comment("get extra info")
      .assign("name_", DocUtils::ToIndexDoc("attrs", plugin->attrs.size()))
      .for_start("i", 1 + plugin->attrs.size(), 1 + plugin->attrs.size() + plugin->inputs.size())
      .func_call("MetaLayout", "MetaLayout layout")
      .call_arg(DocUtils::ToIndexDoc("attrs", "i"))
      .func_call("push_back", "", "layouts_")
      .call_arg("layout")
      .for_end()
      .constructor_end();
  // define serialize
  stack_.comment("serialize method")
      .func_def("serialize", "const std::vector<std::string>")
      .func_start()
      .assign("attrs", "meta_attrs_.Serialize()", "std::vector<std::string>")
      .func_call("push_back", "", "attrs")
      .call_arg("name_")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("push_back", "", "attrs")
      .call_arg(DocUtils::ToAttrAccessDoc(DocUtils::ToIndexDoc("layouts_", "i"), "name()"))
      .for_end()
      .func_end("attrs");
  // malloc outputs/buffers
  stack_.comment("main compute")
      .func_def("compute", "std::vector<torch::Tensor>")
      .func_arg("input_tensors", "const std::vector<torch::Tensor>&")
      .func_start()
      .declare("std::vector<torch::Tensor>", "output_tensors");
  if (plugin->externs.count("infer_buffer")) {
    stack_.declare("std::vector<torch::Tensor>", "buffer_tensors");
  }
  stack_.line()
      .comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("TorchUtils::ToMetaTensor", "MetaTensor m_input")
      .call_arg(DocUtils::ToIndexDoc("input_tensors", "i"))
      .call_arg(DocUtils::ToIndexDoc("layouts_", "i"))
      .func_call("push_back", "", "input_metas")
      .call_arg("m_input")
      .for_end();
  // malloc outputs and buffers
  ICHECK(plugin->externs.count("infer_output")) << "Can not find extern shape";
  CodeGenMalloc(plugin, plugin->outputs, "output");
  if (plugin->externs.count("infer_buffer")) {
    CodeGenMalloc(plugin, plugin->buffers, "buffer");
  }
  // do the compute
  stack_.line().comment("do the compute").cond_if("input_tensors[0].is_cuda()");
  CodeGenCompute(plugin, "cuda");
  stack_.cond_else();
  CodeGenCompute(plugin, "cpu");
  stack_.cond_end();
  stack_.func_end("output_tensors").line();
  stack_.comment("define members")
      .declare(plugin->name + "_attr", "meta_attrs_")
      .declare("std::vector<MetaLayout>", "layouts_")
      .declare("std::string", "name_");
  stack_.struct_end().line();
}

void TorchPluginCodeGen::CodeGenRegister(const Plugin& plugin) {
  stack_.comment("Python wrapper for plugin " + plugin->name)
      .func_def(plugin->name + "_entry", "std::vector<torch::Tensor>")
      .func_arg("instance", "const c10::intrusive_ptr<" + plugin->name + "_op>&");
  for (const auto& input : plugin->inputs) {
    stack_.func_arg(input->name, "const torch::Tensor&");
  }
  for (const auto& attr : plugin->attrs) {
    stack_.func_arg(attr->name, attr->type);
  }
  stack_.func_start().declare("std::vector<torch::Tensor>", "inputs", 0, false);
  for (const auto& input : plugin->inputs) {
    stack_.declare_arg(input->name);
  }
  const auto& outputs_doc = DocUtils::ToDeclareDoc("std::vector<torch::Tensor>", "outputs");
  stack_.func_call("compute", outputs_doc, DocUtils::ToPtrDoc("instance")).call_arg("inputs");
  stack_.func_end("outputs");
  const String& op_doc = plugin->name + "_op";
  stack_.comment("Bind plugin " + plugin->name + " to python")
      .scope_start("TORCH_LIBRARY(" + op_doc + ", m) {")
      .scope_start("m.class_<" + op_doc + ">(\"" + plugin->name + "_op\")")
      .line(".def(torch::init<const std::vector<std::string>>())")
      .line(".def(\"compute\", " + op_doc + "::compute)")
      .scope_start(".def_pickle(")
      .scope_start("[](const c10::intrusive_ptr<" + op_doc + ">& self)")
      .scope_start("-> std::vector<std::string> {")
      .line("return self->serialize();")
      .scope_end()
      .line("},")
      .scope_end()
      .scope_start("[](std::vector<std::string> state)")
      .scope_start("-> c10::intrusive_ptr<" + op_doc + "> {")
      .line("return c10::make_intrusive<" + op_doc + ">(std::move(state));")
      .scope_end()
      .line("}")
      .scope_end()
      .scope_end()
      .line(");")
      .scope_end()
      .func_call("def", "", "m")
      .call_arg(DocUtils::ToStrDoc(plugin->name + "_entry"))
      .call_arg(plugin->name + "_entry")
      .scope_end()
      .line("}");
}

void TorchPluginCodeGen::CodeGenCmake(const std::set<String>& devices) {
  stack_.line("cmake_minimum_required(VERSION " + config()->cmake_version + " FATAL_ERROR)")
      .line("project(msc_torch_plugin)");
  if (devices.count("cuda")) {
    stack_.line("find_package(CUDA)").line("add_definitions(-DPLUGIN_SUPPORT_CUDA)");
  }
  stack_.line("set(CMAKE_CXX_STANDARD 14)")
      .line("list(APPEND CMAKE_PREFIX_PATH \"" + config()->torch_prefix + "\")")
      .line("find_package(Torch REQUIRED)");
  stack_.line("add_definitions(-DPLUGIN_SUPPORT_TORCH)");
  for (const auto& pair : config()->flags) {
    if (pair.second == "") {
      stack_.line("add_definitions(-D" + pair.first + ")");
    } else {
      stack_.line("add_definitions(-D" + pair.first + "=" + pair.second + ")");
    }
  }
  stack_.line("file(GLOB_RECURSE TORCH_CC_SRCS src/*.cc)");
  if (devices.count("cuda")) {
    stack_.line("file(GLOB_RECURSE TORCH_CU_SRCS src/*.cu)");
  }
  if (devices.count("cuda")) {
    stack_.line("cuda_add_library(msc_torch_plugin SHARED ${TORCH_CC_SRCS} ${TORCH_CU_SRCS})");
  } else {
    stack_.line("add_library(msc_torch_plugin SHARED ${TORCH_CC_SRCS})");
  }
  if (config()->includes.size() > 0) {
    String includes = "";
    for (const auto& include : config()->includes) {
      includes = includes + " " + include;
    }
    stack_.line("target_include_directories(msc_torch_plugin PUBLIC" + includes + ")");
  }
  String libs = "";
  for (const auto& lib : config()->libs) {
    libs = libs + " " + lib;
  }
  stack_.line("target_link_libraries(msc_torch_plugin ${TORCH_LIBRARIES}" + libs + ")");
  if (config()->install_dir.size() > 0) {
    stack_.line("SET(LIBRARY_OUTPUT_PATH " + config()->install_dir + ")");
    if (libs.size() > 0) {
      stack_.line("file(COPY " + libs + " DESTINATION " + config()->install_dir + ")");
    }
  }
}

void TorchPluginCodeGen::CodeGenPluginManager(const Plugin& plugin) {}

void TorchPluginCodeGen::CodeGenPluginConvert(const Plugin& plugin) {}

void TorchPluginCodeGen::CodeGenMalloc(const Plugin& plugin, const Array<PluginTensor>& tensors,
                                       const String& collect) {
  Array<String> call_args{"input_metas", "meta_attrs_", "true"};
  stack_.line().comment("malloc " + collect).declare("std::vector<MetaTensor>", collect + "_metas");
  CodeGenSafeCall(plugin->externs["infer_" + collect], call_args, collect + "_metas");
  for (size_t i = 0; i < tensors.size(); i++) {
    const String& var_name = "t_" + tensors[i]->name;
    const String& opt_name = "opt_" + tensors[i]->name;
    const auto& idx_meta = DocUtils::ToIndexDoc(collect + "_metas", i);
    stack_.func_call("torch::TensorOptions", "auto " + opt_name).method_call("dtype");
    int dtype_idx = plugin->FindDtypeRefIdx(tensors[i]);
    if (dtype_idx >= 0) {
      const auto& input_doc = DocUtils::ToIndexDoc("input_tensors", dtype_idx);
      stack_.call_arg(DocUtils::ToAttrAccessDoc(input_doc, "dtype()"));
    } else {
      stack_.call_arg("TorchUtils::ToTorchType(\"" + tensors[i]->dtype + "\")");
    }
    stack_.method_call("device");
    int device_idx = plugin->FindDeviceRefIdx(tensors[i]);
    if (device_idx >= 0) {
      const auto& input_doc = DocUtils::ToIndexDoc("input_tensors", device_idx);
      stack_.call_arg(DocUtils::ToAttrAccessDoc(input_doc, "device()"));
    } else {
      stack_.call_arg("TorchUtils::ToTorchDevice(\"" + tensors[i]->device + "\")");
    }
    stack_.func_call("torch::zeros", "torch::Tensor " + var_name)
        .call_arg(DocUtils::ToAttrAccessDoc(idx_meta, "meta_shape()"))
        .call_arg(opt_name);
    stack_.func_call("push_back", "", collect + "_tensors").call_arg(var_name);
  }
}

void TorchPluginCodeGen::CodeGenCompute(const Plugin& plugin, const String& device) {
  auto prepare_tensor = [this](const PluginTensor& tensor, const Map<String, String>& dtypes,
                               size_t idx, const String& collect) {
    const String& t_name = "d_" + tensor->name;
    const String& t_dtype = dtypes.count(tensor->name) ? dtypes[tensor->name] : tensor->dtype;
    String assign_to = "DataTensor<" + t_dtype + "> " + t_name;
    if (collect == "input") {
      assign_to = "const " + assign_to;
    }
    stack_.func_call("TorchUtils::ToDataTensor<" + t_dtype + ">", assign_to)
        .call_arg(DocUtils::ToIndexDoc(collect + "_tensors", idx))
        .call_arg(DocUtils::ToIndexDoc(collect + "_metas", idx))
        .call_arg(collect == "input" ? "true" : "false");
    return t_name;
  };

  if (plugin->externs.count(device + "_compute")) {
    if (device == "cuda") {
      stack_.declare("cudaStream_t", "stream").assign("stream", "at::cuda::getCurrentCUDAStream()");
    }
    for (const auto& dtypes : plugin->GetDtypeMatrix()) {
      Array<String> compute_args;
      Map<String, String> tensor_dtypes;
      for (const auto& pair : dtypes) {
        const String& ref_dtype = plugin->inputs[pair.first->value]->dtype;
        for (const auto& t : plugin->inputs) {
          if (t->dtype == ref_dtype) {
            tensor_dtypes.Set(t->name, pair.second);
          }
        }
        for (const auto& t : plugin->outputs) {
          if (t->dtype == ref_dtype) {
            tensor_dtypes.Set(t->name, pair.second);
          }
        }
        for (const auto& t : plugin->buffers) {
          if (t->dtype == ref_dtype) {
            tensor_dtypes.Set(t->name, pair.second);
          }
        }
      }
      String dtype_cond = "";
      size_t cnt = 0;
      for (const auto& pair : dtypes) {
        dtype_cond = dtype_cond + "TorchUtils::TypeName(input_tensors[" +
                     std::to_string(pair.first->value) + "].scalar_type())==\"" + pair.second +
                     "\"";
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
      for (size_t i = 0; i < plugin->buffers.size(); i++) {
        const String& t_name = prepare_tensor(plugin->buffers[i], tensor_dtypes, i, "buffer");
        compute_args.push_back(t_name);
      }
      compute_args.push_back("meta_attrs_");
      if (device == "cuda") {
        compute_args.push_back("stream");
      }
      CodeGenSafeCall(plugin->externs[device + "_compute"], compute_args);
      stack_.cond_end();
    }
  } else {
    stack_.comment("skip " + device + " compute");
  }
}

TVM_REGISTER_GLOBAL("msc.plugin.GetTorchPluginSources")
    .set_body_typed([](const String& codegen_config, const String& print_config,
                       const String& codegen_type) -> Map<String, String> {
      TorchPluginCodeGen codegen = TorchPluginCodeGen(codegen_config);
      if (codegen_type == "sources") {
        return codegen.GetPluginSources(print_config);
      }
      if (codegen_type == "manager") {
        return codegen.GetManagerSources(print_config);
      }
      if (codegen_type == "convert") {
        return codegen.GetConvertSources(print_config);
      }
      return Map<String, String>();
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
