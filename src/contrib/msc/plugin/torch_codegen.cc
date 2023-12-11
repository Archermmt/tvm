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
      .func_arg("inputs", "const std::vector<torch::Tensor>&")
      .func_start()
      .declare("std::vector<torch::Tensor>", "output_tensors")
      .declare("std::vector<torch::Tensor>", "buffer_tensors")
      .line()
      .comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("TorchUtils::ToMetaTensor", "MetaTensor m_input")
      .call_arg(DocUtils::ToIndexDoc("inputs", "i"))
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
  // prepare compute datas
  stack_.line()
      .comment("prepare compute datas")
      .func_call("TorchUtils::ToDataTensors", "input_datas")
      .call_arg("inputs")
      .call_arg("input_metas")
      .call_arg("true")
      .func_call("TorchUtils::ToDataTensors", "output_datas")
      .call_arg("output_tensors")
      .call_arg("output_metas")
      .call_arg("false")
      .func_call("TorchUtils::ToDataTensors", "buffer_datas")
      .call_arg("buffer_tensors")
      .call_arg("buffer_metas")
      .call_arg("false");
  // do the compute
  stack_.line().comment("do the compute").cond_if("inputs[0].is_cuda()");
  if (plugin->externs.count("cuda_compute")) {
    stack_.declare("cudaStream_t", "stream").assign("stream", "at::cuda::getCurrentCUDAStream()");
    Array<String> compute_args{"input_datas", "output_datas", "buffer_datas", "meta_attrs_",
                               "stream"};
    CodeGenSafeCall(plugin->externs["cuda_compute"], compute_args);
  } else {
    stack_.comment("skip cuda compute");
  }
  stack_.cond_else();
  if (plugin->externs.count("cpu_compute")) {
    Array<String> compute_args{"input_datas", "output_datas", "buffer_datas", "meta_attrs_"};
    CodeGenSafeCall(plugin->externs["cpu_compute"], compute_args);
  } else {
    stack_.comment("skip cpu compute");
  }
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
      .line("return self->serialize()")
      .scope_end()
      .line("},")
      .scope_end()
      .scope_start("[](std::vector<std::string> state)")
      .scope_start("-> c10::intrusive_ptr<" + op_doc + "> {")
      .line("return c10::make_intrusive<" + op_doc + ">(std::move(state))")
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
  stack_.line("list(APPEND CMAKE_PREFIX_PATH \"" + config()->torch_prefix + "\")")
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

  /*
  .line("find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS " + config()->tensorrt_root +
        " PATH_SUFFIXES include)")
      .line("find_library(TENSORRT_LIB_DIR nvinfer HINTS " + config()->tensorrt_root +
            " PATH_SUFFIXES lib)")
      .line(
          "message(STATUS \"Build project with TENSORRT_INCLUDE_DIR ${TENSORRT_INCLUDE_DIR} and "
          "TENSORRT_LIB_DIR "
          "${TENSORRT_LIB_DIR}\")")
      .line("add_definitions(-DTRT_MAJOR=" + std::to_string(config()->version[0]) + ")")
      .line("add_definitions(-DTRT_MINOR=" + std::to_string(config()->version[1]) + ")")
      .line("add_definitions(-DTRT_PATCH=" + std::to_string(config()->version[2]) + ")")
      .line("file(GLOB_RECURSE TRT_SRCS *.cc)")
      .line("cuda_add_executable(" + graph()->name + " ${TRT_SRCS})")
      .line("target_include_directories(" + graph()->name + " PUBLIC ${TENSORRT_INCLUDE_DIR})")
      .line("target_link_libraries(" + graph()->name + " ${TENSORRT_LIB_DIR})");
      */
}

void TorchPluginCodeGen::CodeGenPluginManager(const Plugin& plugin) {}

void TorchPluginCodeGen::CodeGenPluginConvert(const Plugin& plugin) {}

void TorchPluginCodeGen::CodeGenMalloc(const Plugin& plugin, const Array<PluginTensor>& tensors,
                                       const String& collect) {
  Array<String> call_args{"input_metas", "meta_attrs_", "true"};
  stack_.line().comment("malloc " + collect).declare("std::vector<MetaTensor>", collect + "_metas");
  CodeGenSafeCall(plugin->externs["infer_" + collect], call_args, collect + "_metas");
  for (size_t i = 0; i < tensors.size(); i++) {
    const String& assign_name = "t_" + tensors[i]->name;
    const auto& idx_meta = DocUtils::ToIndexDoc(collect + "_metas", i);
    stack_.func_call("torch::zeros", "torch::Tensor " + assign_name)
        .call_arg(DocUtils::ToAttrAccessDoc(idx_meta, "meta_shape()"));
    int dtype_idx = plugin->FindDtypeRefIdx(tensors[i]);
    if (dtype_idx >= 0) {
      const auto& input_doc = DocUtils::ToIndexDoc("inputs", dtype_idx);
      stack_.call_arg(DocUtils::ToAttrAccessDoc(input_doc, "dtype()"));
    } else {
      stack_.call_arg("TorchUtils::ToTorchType(\"" + tensors[i]->dtype + "\")");
    }
    stack_.method_call("device");
    int device_idx = plugin->FindDeviceRefIdx(tensors[i]);
    if (device_idx >= 0) {
      const auto& input_doc = DocUtils::ToIndexDoc("inputs", device_idx);
      stack_.call_arg(DocUtils::ToAttrAccessDoc(input_doc, "device()"));
    } else {
      stack_.call_arg("TorchUtils::ToTorchDevice(\"" + tensors[i]->device + "\")");
    }
    stack_.func_call("push_back", "", collect + "_tensors").call_arg(assign_name);
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
