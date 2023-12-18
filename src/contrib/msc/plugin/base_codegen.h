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
 * \file src/contrib/msc/plugin/base_codegen.h
 * \brief The codegen for Plugin.
 */
#ifndef TVM_CONTRIB_MSC_PLUGIN_BASE_CODEGEN_H_
#define TVM_CONTRIB_MSC_PLUGIN_BASE_CODEGEN_H_

#include <dmlc/json.h>
#include <tvm/script/printer/doc.h>

#include <set>
#include <string>

#include "../core/codegen/code_stack.h"
#include "../core/printer/cpp_printer.h"
#include "../core/printer/python_printer.h"
#include "plugin.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief CodeGen for Plugin
 */
template <typename ConfigType>
class BasePluginCodeGen {
 public:
  /*!
   * \brief The constructor of BasePluginCodeGen
   * \param config the options for codegen.
   */
  explicit BasePluginCodeGen(const std::string& config = "") {
    config_.reset(new ConfigType());
    if (config.size() > 0) {
      std::istringstream is(config);
      dmlc::JSONReader reader(&is);
      reader.Read(config_.get());
    }
  }

  virtual ~BasePluginCodeGen() = default;

  /*! \brief Get plugin sources*/
  virtual const Map<String, String> GetBuildSources(const std::string& print_options = "") {
    Map<String, String> sources;
    // plugin sources
    for (const auto& name : ListPluginNames()) {
      const auto& plugin = GetPlugin(name);
      // attr declare
      const String& macro = "TVM_CONTRIB_MSC_" + StringUtils::Upper(MetaAttrCls(plugin)) + "_H_";
      this->stack_.line("#ifndef " + macro)
          .line("#define " + macro)
          .line()
          .line("#include \"utils/plugin_utils.h\"")
          .line();
      StartNamespace();
      CodeGenAttrDeclare(plugin);
      EndNamespace();
      this->stack_.line("#endif  // " + macro);
      sources.Set(plugin->name + "_attr.h", ToCppSource(print_options));
      // attr define
      this->stack_.line("#include \"" + plugin->name + "_attr.h\"").line();
      StartNamespace();
      CodeGenAttrDefine(plugin);
      EndNamespace();
      sources.Set(plugin->name + "_attr.cc", ToCppSource(print_options));
      // op define and register
      CodeGenOpHeader(plugin);
      StartNamespace();
      CodeGenOpDefine(plugin);
      CodeGenOpRegister(plugin);
      EndNamespace();
      sources.Set(plugin->name + "_op.cc", ToCppSource(print_options));
      // op runtime
      if (this->config()->with_runtime) {
        CodeGenOpHeader(plugin);
        StartNamespace();
        CodeGenOpRuntime(plugin);
        EndNamespace();
        sources.Set(plugin->name + "_runtime.cc", ToCppSource(print_options));
      }
    }
    // cmakelists
    std::set<String> devices;
    for (const auto& name : ListPluginNames()) {
      const auto& plugin = GetPlugin(name);
      for (const auto& pair : plugin->externs) {
        if (StringUtils::EndsWith(pair.first, "_compute")) {
          devices.insert(StringUtils::Replace(pair.first, "_compute", ""));
        }
      }
    }
    CodeGenCmake(devices);
    sources.Set("CMakeLists.txt", ToCppSource(print_options));
    return sources;
  }

  /*! \brief Get manager sources*/
  virtual const Map<String, String> GetManagerSources(const std::string& print_options = "") {
    Map<String, String> sources;
    this->stack_.comment("Auto generated manager of msc plugin");
    sources.Set("__init__.py", ToPySource(print_options));
    CodeGenManagerImports();
    this->stack_.line();
    CodeGenManagerUtils();
    this->stack_.class_def("PluginManager(object)").class_start();
    CodeGenManagerMethods();
    for (const auto& name : ListPluginNames()) {
      CodeGenPluginManager(GetPlugin(name));
    }
    if (this->config()->need_convert) {
      Map<Plugin, String> symbols;
      this->stack_.func_def("get_converters")
          .func_decorator("classmethod")
          .func_arg("cls", "object")
          .func_start();
      for (const auto& name : ListPluginNames()) {
        const auto& plugin = GetPlugin(name);
        const auto& symbol = CodeGenPluginConvert(plugin);
        symbols.Set(plugin, symbol);
      }
      this->stack_.assign("converters", "{}");
      for (const auto& pair : symbols) {
        this->stack_.assign("converters[\"" + pair.second + "\"]", ConverterName(pair.first));
      }
      this->stack_.func_end("converters");
    }
    this->stack_.class_end();
    sources.Set("manager.py", ToPySource(print_options));
    return sources;
  }

 protected:
  /*! \brief Header of plugin files*/
  virtual void CodeGenOpHeader(const Plugin& plugin) {
    this->stack_.line("#include \"" + plugin->name + "_attr.h\"");
    std::set<String> include_headers;
    for (const auto& pair : plugin->externs) {
      if (pair.second->header.size() > 0 && !include_headers.count(pair.second->header)) {
        this->stack_.line("#include \"" + pair.second->header + "\"");
        include_headers.insert(pair.second->header);
      }
    }
    this->stack_.line();
  }

  /*! \brief Start the namespace*/
  void StartNamespace() {
    this->stack_.line("namespace tvm {").line("namespace contrib {").line("namespace msc {").line();
  }

  /*! \brief End the namespace*/
  void EndNamespace() {
    this->stack_.line("}  // namespace msc")
        .line("}  // namespace contrib")
        .line("}  // namespace tvm");
  }

  /*! \brief Codegen safe call extern*/
  void CodeGenSafeCall(const PluginExtern& extern_func,
                       const Array<String>& call_args = Array<String>(), const String& ret = "") {
    this->stack_.scope_start("try {").func_call(extern_func->name, ret);
    for (const auto& arg : call_args) {
      this->stack_.call_arg(arg);
    }
    this->stack_.scope_end()
        .scope_start("} catch (const std::exception& exc) {")
        .line("std::cerr << \"Failed to run extern " + extern_func->name +
              " : \" << exc.what() << std::endl;")
        .line("throw std::runtime_error(\"Failed to run extern " + extern_func->name + "\");")
        .scope_end()
        .line("}");
  }

  /*! \brief Codegen attr struct declare for plugin*/
  virtual void CodeGenAttrDeclare(const Plugin& plugin) {
    this->stack_.struct_start(MetaAttrCls(plugin)).comment("define attributes");
    for (const auto& attr : plugin->attrs) {
      this->stack_.declare(ToCppType(attr->type), attr->name);
      if (attr->default_value.size() > 0) {
        this->stack_.declare_arg(attr->default_value);
      }
    }
    this->stack_.line()
        .comment("print method")
        .func_def("operator<<", "friend std::ostream&")
        .func_arg("out", "std::ostream&")
        .func_arg("attrs", MetaAttrCls(plugin) + "&")
        .func_start()
        .line("out << \"[" + MetaAttrCls(plugin) + "] : \";");
    for (const auto& attr : plugin->attrs) {
      this->stack_.line("out << \"| " + attr->name + "(" + attr->type + ")=\" << attrs." +
                        attr->name + ";");
    }
    this->stack_.func_end("out").struct_end();
  }

  /*! \brief Get plugin attr define for plugin*/
  virtual void CodeGenAttrDefine(const Plugin& plugin) {}

  /*! \brief Codegen define for plugin*/
  virtual void CodeGenOpDefine(const Plugin& plugin) = 0;

  /*! \brief Codegen register for plugin*/
  virtual void CodeGenOpRegister(const Plugin& plugin) = 0;

  /*! \brief Get plugin runtime source*/
  virtual void CodeGenOpRuntime(const Plugin& plugin) {}

  /*! \brief Codegen cmake file*/
  virtual void CodeGenCmake(const std::set<String>& devices) {}

  /*! \brief Codegen manager utils*/
  virtual void CodeGenManagerUtils() {
    this->stack_.func_def("to_string", "str")
        .func_arg("value", "Any")
        .func_start()
        .switch_start("isinstance(value, (list, tuple))")
        .assign("str_value", "\",\".join([str(len(value))] + [_str_string(v) for v in value])")
        .switch_case("isinstance(value, bool)")
        .assign("str_value", "\"1\" if value else \"0\"")
        .switch_case()
        .assign("str_value", "str(value)")
        .switch_end()
        .func_end("str_value");
  }

  /*! \brief Codegen manager imports*/
  virtual void CodeGenManagerImports() {
    this->stack_.line("import os")
        .line("import shutil")
        .line("import ctypes")
        .line("from typing import Any, List");
  }

  /*! \brief Codegen manager methods*/
  virtual void CodeGenManagerMethods() {
    // copy the libs
    this->stack_.func_def("copy_libs")
        .func_arg("self", "object")
        .func_arg("dst", "str")
        .func_start()
        .cond_if("not os.path.isdir(dst)")
        .func_call("makedirs", "", "os")
        .call_arg("dst")
        .cond_end()
        .for_start("lib", "os.listdir(self._lib_folder)")
        .func_call("shutil.copyfile")
        .call_arg("os.path.join(self._lib_folder, lib)")
        .call_arg("os.path.join(dst, lib)")
        .for_end()
        .func_end();
    // get op names
    this->stack_.func_def("get_op_names", "List[str]")
        .func_arg("self", "object")
        .func_start()
        .assign("names", "[]");
    for (const auto& name : ListPluginNames()) {
      this->stack_.func_call("append", "", "names").call_arg(DocUtils::ToStrDoc(name));
    }
    this->stack_.func_end("names");
    // get ops info
    this->stack_.func_def("get_ops_info", "dict")
        .func_arg("self", "object")
        .func_start()
        .assign("info", "{}");
    for (const auto& name : ListPluginNames()) {
      ICHECK(this->config()->ops_info.count(name)) << "Can not find op info for " << name;
      const auto& info = this->config()->ops_info[name];
      this->stack_.assign(DocUtils::ToIndexDoc("info", DocUtils::ToStrDoc(name)), info);
    }
    this->stack_.func_end("info");
  };

  /*! \brief Codegen manager for plugin*/
  virtual void CodeGenPluginManager(const Plugin& plugin) = 0;

  /*! \brief Codegen convert function for plugin*/
  virtual const String CodeGenPluginConvert(const Plugin& plugin) { return plugin->name; }

  /*! \brief Change code stack to cpp source*/
  const String ToCppSource(const std::string& print_options = "") {
    CppPrinter printer(print_options);
    for (const auto& d : this->stack_.GetDocs()) {
      printer.Append(d);
    }
    this->stack_.Reset();
    return printer.GetString();
  }

  /*! \brief Change code stack to python source*/
  const String ToPySource(const std::string& print_options = "") {
    PythonPrinter printer(print_options);
    for (const auto& d : this->stack_.GetDocs()) {
      printer.Append(d);
    }
    this->stack_.Reset();
    return printer.GetString();
  }

  const Map<String, String> GetTensorDtypes(const Plugin& plugin,
                                            const Map<Integer, String>& dtypes) {
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
    return tensor_dtypes;
  }

  /*! \brief Change plugin comment in python*/
  const String GetPyComment(const Plugin& plugin) {
    String comment = "Python wrapper for " + plugin->name + "\nInputs\n------";
    for (const auto& t : plugin->inputs) {
      comment = comment + "\n" + t->name + ": " + t->dtype + "\n  " + t->describe;
    }
    comment = comment + "\nOutputs\n-------";
    for (const auto& t : plugin->outputs) {
      comment = comment + "\n" + t->name + ": " + t->dtype + "\n  " + t->describe;
    }
    if (plugin->attrs.size() > 0) {
      comment = comment + "\nAttributes\n-----------";
      for (const auto& a : plugin->attrs) {
        comment = comment + "\n" + a->name + ": " + ToPyType(a->type) + "\n  " + a->describe;
      }
    }
    return comment;
  }

  /*! \brief Get class name for meta attrs*/
  const String MetaAttrCls(const Plugin& plugin) const { return plugin->name + "MetaAttrs"; }

  /*! \brief Get converter name for plugin*/
  const String ConverterName(const Plugin& plugin) const { return plugin->name + "Converter"; }

  /*! \brief Check if the type is list type. */
  bool IsListType(const String& type) { return StringUtils::StartsWith(type, "list"); }

  /*! \brief Get type of element. */
  const String GetEleType(const String& type) {
    if (!IsListType(type)) {
      return "";
    }
    return StringUtils::Replace(StringUtils::Replace(type, "list(", ""), ")", "");
  }

  /*! \brief Type name in cpp*/
  virtual const String ToCppType(const String& type) {
    if (IsListType(type)) {
      const auto& ele_type = GetEleType(type);
      return "std::vector<" + ToCppType(ele_type) + ">";
    }
    if (type == "int64") {
      return "int64_t";
    }
    if (type == "int32" || type == "int") {
      return "int32_t";
    }
    if (type == "int8") {
      return "int8_t";
    }
    if (type == "string") {
      return "std::string";
    }
    return type;
  }

  /*! \brief Type name in python*/
  virtual const String ToPyType(const String& type) {
    if (IsListType(type)) {
      const auto& ele_type = GetEleType(type);
      return "List[" + ToPyType(ele_type) + "]";
    }
    if (type == "int64" || type == "int32" || type == "int" || type == "int8") {
      return "int";
    }
    if (type == "string") {
      return "str";
    }
    return type;
  }

  /*!
   * \brief Compare version with version in config
   * 0 for same version, 1 for greater version, -1 for less version
   */
  int CompareVersion(size_t major, size_t minor, size_t patch) {
    return CommonUtils::CompareVersion(this->config()->version, {major, minor, patch});
  }

  /*! \brief The config of plugin codegen*/
  const std::shared_ptr<ConfigType> config() { return config_; }

  /*! \brief The stack of codes*/
  CodeStack stack_;

 private:
  std::shared_ptr<ConfigType> config_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_PLUGIN_BASE_CODEGEN_H_
