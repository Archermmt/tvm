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
 * \file src/contrib/msc/framework/tensorrt/codegen.cc
 * \brief Codegen related classes.
 */

#include "codegen.h"

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::relax;

void TensorRTCodeGen::CodeGenClassDeclare() {
  stack_.line("#include \"NvInfer.h\"")
      .line("#include \"NvInferRuntimeCommon.h\"")
      .line("#include \"utils/base.h\"")
      .line()
      .line("using namsespace nvinfer1;")
      .line();
  StartNamespace();
  stack_.class_def(graph()->name).class_start().scope_start("public:");
  // declare build method
  stack_.func_def("Build", "bool")
      .func_arg("builder", "TRTPtr<IBuilder>&")
      .func_arg("network", "TRTPtr<INetworkDefinition>&");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.func_arg("config", "TRTPtr<IBuilderConfig>&");
  }
  stack_.func_arg("logger", "TRTLogger&").func_start().func_end();
  // declare test method
  stack_.func_def("Test", "bool")
      .func_arg("engine", "TRTPtr<ICudaEngine>&")
      .func_arg("reader", "DatasetReader&")
      .func_arg("test_iter", "size_t")
      .func_arg("logger", "TRTLogger&")
      .func_start()
      .func_end();
  // define cleanup method
  stack_.func_def("CleanUp", "bool")
      .func_start()
      .for_start("mem", "mWeights")
      .call_start("free")
      .call_arg("(void*) (mem.second.values)")
      .call_end()
      .for_end()
      .func_end("true");
  // end public scope
  stack_.scope_end();
  // private scope
  stack_.scope_start("private:").declare("std::map<std::string, Weights>", "mWeights").scope_end();
  stack_.class_end();
  EndNamespace();
}

void TensorRTCodeGen::CodeGenClassDefine() {
  auto malloc_buffer = [this](const MSCTensor& tensor) {
    const String& idx_var = "idx_" + tensor->alias;
    this->stack_.call_start("engine->getBindingIndex")
        .call_arg(DocUtils::ToStrDoc(tensor->alias))
        .call_end("const int " + idx_var)
        .call_start("CHECK")
        .call_inplace_start("cudaMalloc")
        .call_arg("&gpu_buffers[" + idx_var + "]")
        .call_arg(GetTensorBytes(tensor))
        .call_inplace_end()
        .call_end()
        .call_start("malloc")
        .call_arg(GetTensorBytes(tensor))
        .call_end("cpu_buffers[" + idx_var + "]");
  };

  stack_.line("#include \"" + graph()->name + ".h\"").line();
  StartNamespace();
  // start define build method
  stack_.func_def(graph()->name + "::Build", "bool")
      .func_arg("builder", "TRTPtr<IBuilder>&")
      .func_arg("network", "TRTPtr<INetworkDefinition>&");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.func_arg("config", "TRTPtr<IBuilderConfig>&");
  }
  stack_.func_arg("logger", "TRTLogger&").func_start();
  if (graph()->weight_holders.size() > 0) {
    stack_.assign("mWeights", "TRTUtils::LoadWeights(\"" + graph()->name + ".wts\")");
  }
  // build layers
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    for (const auto& d : GetOpCodes(node)) {
      stack_.line(d);
    }
  }
  // end define build method
  stack_.func_end("true");
  // start define test method
  stack_.func_def(graph()->name + "::Test", "bool")
      .func_arg("engine", "TRTPtr<ICudaEngine>&")
      .func_arg("reader", "DatasetReader&")
      .func_arg("test_iter", "size_t")
      .func_arg("logger", "TRTLogger&")
      .func_start();
  stack_.comment("Create context")
      .call_start("TRTPTr<IExecutionContext>")
      .call_inplace_start("engine->createExecutionContext")
      .call_inplace_end()
      .call_end("auto context");
  ReturnOnFail("context", "Failed to create the context");
  // prepare variables
  stack_.declare("size_t", "cnt", 0, false)
      .declare_arg(0)
      .declare("bool", "pass", 0, false)
      .declare_arg("true")
      .declare("cudaStream_t", "stream")
      .call_start("ICHECK")
      .call_inplace_start("cudaStreamCreate")
      .call_arg("&stream")
      .call_inplace_end()
      .call_end();
  // malloc buffers
  size_t binding_num = graph()->input_names.size() + graph()->output_names.size();
  stack_.comment("Malloc and copy the buffers")
      .declare("void*", "cpu_buffers", binding_num)
      .declare("void*", "gpu_buffers", binding_num);
  for (const auto& i : graph()->GetInputs()) {
    malloc_buffer(i);
  }
  for (const auto& o : graph()->GetOutputs()) {
    malloc_buffer(o);
    stack_.declare(CppDType(o->dtype), "output_" + o->alias,
                   static_cast<size_t>(o->GetSize()->value));
  }
  // read and test datas
  stack_.comment("Read and test datas")
      .while_start("reader.ReadNext(cpu_buffers) && cnt < test_iter")
      .comment("Memcopy inputs host to device");
  // copy inputs
  for (const auto& i : graph()->GetInputs()) {
    stack_.call_start("CHECK")
        .call_inplace_start("cudaMemcpyAsync")
        .call_arg("gpu_buffers[idx_" + i->alias + "]")
        .call_arg("cpu_buffers[idx_" + i->alias + "]")
        .call_arg(GetTensorBytes(i))
        .call_arg("cudaMemcpyHostToDevice")
        .call_arg("stream")
        .call_inplace_end()
        .call_end();
  }
  // enqueue
  stack_.call_start("cudaStreamSynchronize")
      .call_arg("stream")
      .call_end()
      .comment("enquque with gpu buffers")
      .call_start("context->enqueueV2")
      .call_arg("gpu_buffers")
      .call_arg("stream")
      .call_arg("nullptr")
      .call_end()
      .comment("Memcopy outputs device to host");
  // copy outputs
  for (const auto& o : graph()->GetOutputs()) {
    stack_.call_start("CHECK")
        .call_inplace_start("cudaMemcpyAsync")
        .call_arg("output_" + o->alias)
        .call_arg("gpu_buffers[idx_" + o->alias + "]")
        .call_arg(GetTensorBytes(o))
        .call_arg("cudaMemcpyDeviceToHost")
        .call_arg("stream")
        .call_inplace_end()
        .call_end();
  }
  stack_.call_start("cudaStreamSynchronize").call_arg("stream").call_end();
  // compare outputs
  for (const auto& o : graph()->GetOutputs()) {
    stack_.call_start("CompareBuffers")
        .call_arg("(" + CppDType(o->dtype) + "*)cpu_buffers[idx_" + o->alias + "]")
        .call_arg("output_" + o->alias)
        .call_end("pass");
    ReturnOnFail("pass", "Failed to test the output " + o->alias);
  }
  stack_.while_end();
  // clean up
  stack_.comment("Clean up the buffers and stream")
      .call_start("cudaStreamDestory")
      .call_arg("stream")
      .call_end()
      .for_start("i", 0, binding_num)
      .call_start("CHECK")
      .call_inplace_start("cudaFree")
      .call_arg("gpu_buffers[i]")
      .call_inplace_end()
      .call_end()
      .call_start("free")
      .call_arg("cpu_buffers[i]")
      .call_end()
      .for_end();
  // end define test method
  stack_.func_end("true");
  EndNamespace();
}

void TensorRTCodeGen::CodeGenMain() {
  stack_.line("#include \"" + graph()->name + ".h\"")
      .line("#include \"utils/trt_common.h\"")
      .line("#include \"utils/base.h\"")
      .line()
      .line("using namsespace nvinfer1;")
      .line("using namsespace tvm::contrib::msc;")
      .line()
      .func_def("main", "int")
      .func_arg("argc", "int")
      .func_arg("argv", "**char")
      .func_start()
      .declare("TRTLogger", "logger")
      .call_start("logger.setLogServerity");
  if (config()->log_level == 0) {
    stack_.call_arg("ILogger::Serverity::kVERBOSE");
  } else if (config()->log_level == 1) {
    stack_.call_arg("ILogger::Serverity::kINFO");
  } else {
    stack_.call_arg("ILogger::Serverity::kWARNING");
  }
  stack_.call_end();
  // prepare for build
  stack_.comment("Define arguments")
      .assign("pass", "true", "bool")
      .assign("repeat_num", "1000", "int")
      .assign("profile_level", std::to_string(config()->profile_level), "int")
      .cond_if("argc > 1")
      .assign("profile_level", "atoi(argv[1])")
      .cond_end();
  // start build the engine
  stack_.comment("Build engine if not exist")
      .cond_if("!FileUtils::FileExist(\"" + graph()->name + ".trt\")");
  // create builder
  stack_.comment("Create TensorRT tools")
      .call_start("TRTPtr<IBuilder>")
      .call_inplace_start("createInferBuilder")
      .call_arg("logger")
      .call_inplace_end()
      .call_end("auto builder");
  ReturnOnFail("builder", "Failed to create builder");
  // create network
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.assign("flags", "0", "uint32_t")
        .call_start("TRTPtr<INetworkDefinition>")
        .call_inplace_start("builder->createNetworkV2")
        .call_arg("flags")
        .call_inplace_end()
        .call_end("auto network");
  } else {
    stack_.call_start("TRTPtr<INetworkDefinition>")
        .call_inplace_start("builder->createNetwork")
        .call_inplace_end()
        .call_end("auto network");
  }
  ReturnOnFail("network", "Failed to create network");
  // create config
  stack_.call_start("TRTPtr<IBuilderConfig>")
      .call_inplace_start("builder->createBuilderConfig")
      .call_inplace_end()
      .call_end("auto config");
  ReturnOnFail("config", "Failed to create config");
  // build model
  stack_.comment("Build model")
      .declare(graph()->name, "model")
      .call_start("model.Build")
      .call_arg("builder")
      .call_arg("network");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.call_arg("config");
  }
  stack_.call_arg("logger").call_end("pass");
  ReturnOnFail("pass", "Failed to build model");
  // Set profile flag
  stack_.comment("Set profile flag")
      .declare("ProfilingVerbosity", "profile_verbose")
      .cond_if("profile_level == 2")
      .assign("profile_level", "ProfilingVerbosity::kDETAILED")
      .cond_else()
      .cond_if("profile_level == 1")
      .assign("profile_level", "ProfilingVerbosity::kLAYER_NAMES_ONLY")
      .cond_else()
      .assign("profile_level", "ProfilingVerbosity::kNONE")
      .cond_end()
      .cond_end()
      .call_start("config->setProfilingVerbosity")
      .call_arg("profile_verbose")
      .call_end();
  // Serialize engine
  stack_.comment("Serialize engine")
      .call_start("TRTUtils::SerializeEngineToFile")
      .call_arg(DocUtils::ToStrDoc(graph()->name + ".trt"))
      .call_arg("builder")
      .call_arg("network");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.call_arg("config");
  }
  stack_.call_arg("logger").call_end("pass");
  ReturnOnFail("pass", "Failed to serialize the engine");
  // end build the engine
  stack_.cond_end();
  // deserialize the engine
  stack_.comment("Deserialize engine")
      .declare("std::shared_ptr<ICudaEngine>", "engine")
      .call_start("DeserializeEngineFromFile")
      .call_arg(DocUtils::ToStrDoc(graph()->name + ".trt"))
      .call_arg("engine")
      .call_arg("logger")
      .call_end("pass");
  ReturnOnFail("pass", "Failed to deserialize the engine");
  // start profile the engine
  stack_.cond_if("profile_level > 0");
  // dump info by inspector
  stack_.comment("Dump info by inspector")
      .call_start("TRTPtr<IEngineInspector>")
      .call_inplace_start("engine->createEngineInspector")
      .call_inplace_end()
      .call_end("auto inspector")
      .call_start("inspector->getEngineInformation")
      .call_arg("LayerInformation::kJSON")
      .call_end("std::string result")
      .declare("std::ofstream", "os")
      .declare_arg(DocUtils::ToStrDoc(graph()->name + "_info.json"))
      .declare_arg("std::ofstream::trunc")
      .line("os << result << std::flush;");
  // end profile the engine
  stack_.cond_end();
  // test engine
  if (config()->test_iter > 0) {
    stack_.comment("Prepare dataset")
        .declare("DatasetReader", "reader")
        .declare_arg(DocUtils::ToStrDoc(config()->dataset));
    stack_.comment("Test engine by datas")
        .call_start("model.Test")
        .call_arg("engine")
        .call_arg("reader")
        .call_arg(config()->test_iter)
        .call_arg("logger")
        .call_end("pass");
  }
  ReturnOnFail("pass", "Failed to test the engine");
  stack_.func_end("pass ? 0 : 1");
}

void TensorRTCodeGen::CodeGenCmake() {
  stack_.line("cmake_minimum_required(VERSION " + config()->cmake_version + " FATAL_ERROR)")
      .line("project(" + graph()->name + ")")
      .line("find_package(CUDA)")
      .line("find_path(TRT_INCLUDE NvInfer.h HINTS " + config()->tensorrt_root +
            " PATH_SUFFIXES include)")
      .line("find_path(TRT_LIB NvInfer.h HINTS " + config()->tensorrt_root + " PATH_SUFFIXES lib)")
      .line(
          "message(STATUS \"Build project with TRT_INCLUDE ${TRT_INCLUDE} and TRT_LIB "
          "${TRT_LIB}\")")
      .line("add_definitions(-DTRT_MAJOR=" + std::to_string(config()->version[0]) + ")")
      .line("add_definitions(-DTRT_MINOR=" + std::to_string(config()->version[1]) + ")")
      .line("add_definitions(-DTRT_PATCH=" + std::to_string(config()->version[2]) + ")")
      .line("file(GLOB_RECURSE TRT_SRCS *.cc)")
      .line("cuda_add_executable(" + graph()->name + " ${TRT_SRCS})")
      .line("target_include_directories(" + graph()->name + " ${TRT_INCLUDE})")
      .line("target_link_libraries(" + graph()->name + " ${TRT_LIB})");
}

const String TensorRTCodeGen::CppDType(const DataType& dtype) {
  const String& dtype_name = CppCodeGen<TensorRTCodeGenConfig>::DType(dtype);
  if (dtype_name == "int32") {
    return "int";
  }
  if (dtype_name == "int64") {
    return "int64_t";
  }
  if (dtype_name == "float32") {
    return "float";
  }
  if (dtype_name == "float64") {
    return "double";
  }
  return dtype_name;
}

const String TensorRTCodeGen::GetTensorBytes(const MSCTensor& tensor) {
  return std::to_string(tensor->GetSize()->value) + " * sizeof(" + CppDType(tensor->dtype) + ")";
}

void TensorRTCodeGen::ReturnOnFail(const String& flag, const String& err) {
  stack_.cond_if("!" + flag)
      .call_start("logger.log")
      .call_arg("ILogger::Serverity::kERROR")
      .call_arg(DocUtils::ToStrDoc(err))
      .call_end()
      .line("return -1;")
      .cond_end();
};

const Array<Doc> TensorRTCodeGen::GetOpCodes(const MSCJoint& node) {
  const auto& ops_map = GetTensorRTOpCodes();
  auto it = ops_map->find(node->optype);
  ICHECK(it != ops_map->end()) << "Unsupported tensorrt op(" << node->optype << "): " << node;
  it->second->Config(node, config());
  try {
    return it->second->GetDocs();
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to get docs for " << node << " : " << err.message();
    throw err;
  }
}

TVM_REGISTER_GLOBAL("msc.framework.tensorrt.GetTensorRTSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String print_config) -> Map<String, String> {
      TensorRTCodeGen codegen = TensorRTCodeGen(graph, codegen_config);
      return codegen.GetSources(print_config);
    });

/*!
 * \brief Create runtime modules for TensorRT.
 * \param functions The extern functions to be compiled via TensorRT
 * \return Runtime modules.
 */
Array<runtime::Module> TensorRTCompiler(Array<Function> functions,
                                        Map<String, ObjectRef> /*unused*/,
                                        Map<Constant, String> constant_names) {
  Array<runtime::Module> compiled_functions;
  for (const auto& func : functions) {
    std::cout << "[TMINFO] processing " << func << std::endl;
  }
  return compiled_functions;
}

TVM_REGISTER_GLOBAL("relax.ext.msc_tensorrt").set_body_typed(TensorRTCompiler);

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
