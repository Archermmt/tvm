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

void TorchPluginCodeGen::CodeGenAttrDeclare(const Plugin& plugin) {
  BasePluginCodeGen<TorchPluginCodeGenConfig>::CodeGenAttrDeclare(plugin);
  const auto& attr_name = MetaAttrCls(plugin);
  // serialize method for attr
  stack_.comment("serialize method")
      .func_def(attr_name + "_serialize", "std::vector<std::string>")
      .func_arg("meta_attrs", "const " + attr_name + "&");
  // deserialize method for attr
  stack_.comment("deserialize method")
      .func_def(attr_name + "_deserialize")
      .func_arg("attrs", "const std::vector<std::string>&")
      .func_arg("meta_attrs", attr_name + "&");
}

void TorchPluginCodeGen::CodeGenAttrDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  // serialize method for attr
  stack_.func_def(attr_name + "_serialize", "std::vector<std::string>")
      .func_arg("meta_attrs", "const " + attr_name + "&")
      .func_start()
      .declare("std::vector<std::string>", "attrs");
  for (const auto& attr : plugin->attrs) {
    stack_
        .func_call("SerializeUtils::ToString",
                   DocUtils::ToDeclareDoc("std::string", "str_" + attr->name))
        .call_arg(DocUtils::ToAttrAccessDoc("meta_attrs", attr->name))
        .func_call("push_back", "", "attrs")
        .call_arg("str_" + attr->name);
  }
  stack_.func_end("attrs");
  // deserialize method for attr
  stack_.func_def(attr_name + "_deserialize")
      .func_arg("attrs", "const std::vector<std::string>&")
      .func_arg("meta_attrs", attr_name + "&")
      .func_start();
  for (size_t i = 0; i < plugin->attrs.size(); i++) {
    stack_.func_call("SerializeUtils::FromString")
        .call_arg(DocUtils::ToIndexDoc("attrs", i))
        .call_arg(DocUtils::ToAttrAccessDoc("meta_attrs", plugin->attrs[i]->name));
  }
  stack_.func_end();
}

void TorchPluginCodeGen::CodeGenOpDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  stack_.struct_start(plugin->name + " : torch::CustomClassHolder");
  // define constructor
  stack_.constructor_def(plugin->name)
      .constructor_arg("attrs", "const std::vector<std::string>&")
      .constructor_start()
      .comment("get attributes")
      .func_call(attr_name + "_deserialize")
      .call_arg("attrs")
      .call_arg("meta_attrs_")
      .comment("get extra info")
      .assign("name_", DocUtils::ToIndexDoc("attrs", plugin->attrs.size()))
      .for_start("i", 1 + plugin->attrs.size(), 1 + plugin->attrs.size() + plugin->inputs.size())
      .func_call("MetaLayout", DocUtils::ToDeclareDoc("MetaLayout", "layout"))
      .call_arg(DocUtils::ToIndexDoc("attrs", "i"))
      .func_call("push_back", "", "layouts_")
      .call_arg("layout")
      .for_end()
      .constructor_end();
  // define serialize
  stack_.comment("serialize method")
      .func_def("serialize", "const std::vector<std::string>")
      .func_start()
      .assign("attrs", attr_name + "_serialize(meta_attrs_)", "std::vector<std::string>")
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
      .func_call("TorchUtils::ToMetaTensor", DocUtils::ToDeclareDoc("MetaTensor", "m_input"))
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
  String device_cond = "";
  for (size_t i = 0; i < plugin->inputs.size(); i++) {
    if (plugin->inputs[i]->device == "cuda" || plugin->inputs[i]->device == "default") {
      device_cond = device_cond + "input_tensors[" + std::to_string(i) + "].is_cuda()";
    } else {
      device_cond = device_cond + "!input_tensors[" + std::to_string(i) + "].is_cuda()";
    }
    device_cond = device_cond + (i == plugin->inputs.size() - 1 ? "" : " && ");
  }
  stack_.line().comment("do the compute").cond_if(device_cond);
  CodeGenCompute(plugin, "cuda");
  stack_.cond_else();
  CodeGenCompute(plugin, "cpu");
  stack_.cond_end();
  stack_.func_end("output_tensors").line();
  stack_.comment("define members")
      .declare(MetaAttrCls(plugin), "meta_attrs_")
      .declare("std::vector<MetaLayout>", "layouts_")
      .declare("std::string", "name_")
      .line();
  stack_.struct_end();
}

void TorchPluginCodeGen::CodeGenOpRegister(const Plugin& plugin) {
  const auto& entry_name = EntryName(plugin);
  stack_.comment("Python wrapper for plugin " + plugin->name)
      .func_def(entry_name, "std::vector<torch::Tensor>")
      .func_arg("instance", "const c10::intrusive_ptr<" + plugin->name + ">&");
  for (const auto& input : plugin->inputs) {
    stack_.func_arg(input->name, "const torch::Tensor&");
  }
  stack_.func_start().declare("std::vector<torch::Tensor>", "inputs", 0, false);
  for (const auto& input : plugin->inputs) {
    stack_.declare_arg(input->name);
  }
  const auto& outputs_doc = DocUtils::ToDeclareDoc("std::vector<torch::Tensor>", "outputs");
  stack_.func_call("compute", outputs_doc, DocUtils::ToPtrDoc("instance")).call_arg("inputs");
  stack_.func_end("outputs");
  stack_.comment("Bind plugin " + plugin->name + " to python")
      .func_def("TORCH_LIBRARY", DocSymbol::Empty())
      .func_arg(plugin->name, DocSymbol::Empty())
      .func_arg("m", DocSymbol::Empty())
      .func_start()
      .lambda_def("serialize")
      .lambda_arg("op", "const c10::intrusive_ptr<" + plugin->name + ">&")
      .lambda_start()
      .lambda_end(DocUtils::ToAttrAccessDoc(DocUtils::ToPtrDoc("op"), "serialize()"))
      .lambda_def("deserialize")
      .lambda_arg("state", "std::vector<std::string>")
      .lambda_start()
      .lambda_end("c10::make_intrusive<" + plugin->name + ">(std::move(state))")
      .func_call("class_<" + plugin->name + ">", "", "m")
      .call_arg(DocUtils::ToStrDoc(plugin->name))
      .method_call("def", true)
      .call_arg("torch::init<const std::vector<std::string>>()")
      .method_call("def", true)
      .call_arg(DocUtils::ToStrDoc("compute"))
      .call_arg("&" + plugin->name + "::compute")
      .method_call("def_pickle", true)
      .call_arg("serialize")
      .call_arg("deserialize")
      .func_call("def", "", "m")
      .call_arg(DocUtils::ToStrDoc(entry_name))
      .call_arg(entry_name)
      .func_end();
}

void TorchPluginCodeGen::CodeGenCmake(const std::set<String>& devices) {
  stack_.line("cmake_minimum_required(VERSION " + config()->cmake_version + " FATAL_ERROR)")
      .line("project(msc_torch_plugin)");
  if (devices.count("cuda")) {
    stack_.line("find_package(CUDA)").line("add_definitions(-DPLUGIN_ENABLE_CUDA)");
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
    stack_.line("target_include_directories(msc_torch_plugin PUBLIC " + includes + ")");
  }
  String libs = "";
  for (const auto& lib : config()->libs) {
    libs = libs + " " + lib;
  }
  stack_.line("target_link_libraries(msc_torch_plugin ${TORCH_LIBRARIES}" + libs + ")");
  if (config()->install_dir.size() > 0) {
    stack_.line("SET(LIBRARY_OUTPUT_PATH " + config()->install_dir + ")");
    if (config()->libs.size() > 0) {
      stack_.line("file(COPY " + libs + " DESTINATION " + config()->install_dir + ")");
    }
  }
}

void TorchPluginCodeGen::CodeGenManagerImports() {
  stack_.line("import torch");
  BasePluginCodeGen<TorchPluginCodeGenConfig>::CodeGenManagerImports();
}

void TorchPluginCodeGen::CodeGenManagerMethods() {
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
      .cond_if("\"msc_torch_plugin\" in lib")
      .func_call("load_library", "", "torch.classes")
      .call_arg("lib_file")
      .cond_else()
      .func_call("CDLL", "", "ctypes")
      .call_arg("lib_file")
      .cond_end()
      .for_end()
      .func_end();
  BasePluginCodeGen<TorchPluginCodeGenConfig>::CodeGenManagerMethods();
}

void TorchPluginCodeGen::CodeGenPluginManager(const Plugin& plugin) {
  const auto& entry_name = EntryName(plugin);
  stack_.func_def(plugin->name).func_arg("self", "object");
  for (const auto& attr : plugin->attrs) {
    stack_.func_arg(attr->name, attr->type, attr->default_value);
  }
  stack_.func_arg("name", "str", "\"" + plugin->name + "\"")
      .func_arg("layouts", "List[str]", "None")
      .func_start()
      .line()
      .class_def(plugin->name + "(torch.nn.Module)")
      .class_start();
  // init method
  stack_.func_def("__init__").func_arg("self", "torch.nn.Module");
  for (const auto& attr : plugin->attrs) {
    stack_.func_arg(attr->name, attr->type, attr->default_value);
  }
  stack_.func_arg("name", "str", "\"" + plugin->name + "\"")
      .func_arg("layouts", "List[str]", "None")
      .func_start()
      .func_call("__init__", "", "super()");
  for (const auto& attr : plugin->attrs) {
    stack_.assign(DocUtils::ToAttrAccessDoc("self", attr->name), attr->name);
  }
  stack_.assign(DocUtils::ToAttrAccessDoc("self", "name"), "name")
      .cond_if("layouts is None")
      .assign(DocUtils::ToAttrAccessDoc("self", "layouts"),
              "[\"\"] * " + std::to_string(plugin->inputs.size()))
      .cond_else()
      .assign(DocUtils::ToAttrAccessDoc("self", "layouts"), "layouts")
      .cond_end()
      .line()
      .assign("attr_strs", "[]");
  for (const auto& attr : plugin->attrs) {
    stack_.func_call("append", "", "attr_strs").call_arg("to_string(" + attr->name + ")");
  }
  stack_.func_call("append", "", "attr_strs")
      .call_arg("name")
      .func_call("extend", "", "attr_strs")
      .call_arg(DocUtils::ToAttrAccessDoc("self", "layouts"))
      .line()
      .func_call(plugin->name + "." + plugin->name, "self._inner_class", "torch.classes")
      .call_arg("attr_strs")
      .func_end();
  // forward method
  stack_.func_def("forward", "List[torch.Tensor]").func_arg("self", "torch.nn.Module");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "torch.Tensor");
  }
  stack_.func_start()
      .func_call(plugin->name + "." + entry_name, "outputs", "torch.ops")
      .call_arg("self._inner_class");
  for (const auto& t : plugin->inputs) {
    stack_.call_arg(t->name);
  }
  if (plugin->outputs.size() == 1) {
    stack_.func_end(DocUtils::ToIndexDoc("outputs", 0));
  } else {
    stack_.func_end("outputs");
  }
  // end of inner class
  stack_.class_end();
  stack_.func_call(plugin->name, "op");
  for (const auto& attr : plugin->attrs) {
    stack_.call_arg(attr->name);
  }
  stack_.call_arg("name").call_arg("layouts").func_end("op").comment(GetPyComment(plugin), true);
}

const String TorchPluginCodeGen::CodeGenPluginConvert(const Plugin& plugin) {
  stack_.func_def(ConverterName(plugin)).func_start().func_end();
  return plugin->name + "::" + EntryName(plugin);
}

void TorchPluginCodeGen::CodeGenMalloc(const Plugin& plugin, const Array<PluginTensor>& tensors,
                                       const String& collect) {
  Array<String> call_args{"input_metas", "meta_attrs_", "true"};
  stack_.line().comment("malloc " + collect).declare("std::vector<MetaTensor>", collect + "_metas");
  CodeGenSafeCall(plugin->externs["infer_" + collect], call_args, collect + "_metas");
  for (size_t i = 0; i < tensors.size(); i++) {
    const auto& var_name = "t_" + tensors[i]->name;
    stack_.func_call("TorchUtils::MallocTorchTensor", DocUtils::ToDeclareDoc("auto", var_name))
        .call_arg(DocUtils::ToIndexDoc(collect + "_metas", i));
    int device_idx = plugin->FindDeviceRefIdx(tensors[i]);
    if (device_idx >= 0) {
      const auto& input_doc = DocUtils::ToIndexDoc("input_tensors", device_idx);
      stack_.call_arg(DocUtils::ToAttrAccessDoc(input_doc, "device()"));
    } else {
      stack_.call_arg("TorchUtils::ToTorchDevice(\"" + tensors[i]->device + "\")");
    }
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
    for (const auto& dtypes : plugin->GetDtypeMatrix()) {
      Array<String> compute_args;
      const auto& tensor_dtypes = GetTensorDtypes(plugin, dtypes);
      String dtype_cond = "";
      size_t cnt = 0;
      for (const auto& pair : dtypes) {
        dtype_cond = dtype_cond + "input_metas[" + std::to_string(pair.first->value) +
                     "].data_type() == DataUtils::ToMetaType(\"" + pair.second + "\")";
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
        stack_.func_call("at::cuda::getCurrentCUDAStream",
                         DocUtils::ToDeclareDoc("cudaStream_t", "stream"));
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
