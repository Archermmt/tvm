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
  virtual const Map<String, String> GetPluginSources(const std::string& print_options = "") {
    Map<String, String> sources;
    // plugin sources
    for (const auto& name : ListPluginNames()) {
      const auto& plugin = GetPlugin(name);
      CodeGenPluginAttr(plugin);
      sources.Set(plugin->name + "_attr.h", ToSource(print_options));
      CodeGenPluginSource(plugin);
      sources.Set(plugin->name + "_op.cc", ToSource(print_options));
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
    sources.Set("CMakeLists.txt", ToSource(print_options));
    return sources;
  }

  /*! \brief Get manager sources*/
  virtual const Map<String, String> GetManagerSources(const std::string& print_options = "") {
    Map<String, String> sources;
    for (const auto& name : ListPluginNames()) {
      const auto& plugin = GetPlugin(name);
      CodeGenPluginManager(plugin);
    }
    sources.Set("manager.py", ToSource(print_options));
    return sources;
  }

  /*! \brief Get convert sources*/
  virtual const Map<String, String> GetConvertSources(const std::string& print_options = "") {
    Map<String, String> sources;
    for (const auto& name : ListPluginNames()) {
      const auto& plugin = GetPlugin(name);
      CodeGenPluginConvert(plugin);
    }
    sources.Set("convert.py", ToSource(print_options));
    return sources;
  }

 protected:
  /*! \brief Header of plugin files*/
  virtual void CodeGenHeader(const Plugin& plugin) {
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
    this->stack_.line()
        .line("} // namespace msc")
        .line("} // namespace contrib")
        .line("} // namespace tvm");
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

  /*! \brief Get plugin attr*/
  virtual void CodeGenPluginAttr(const Plugin& plugin) {
    const String& macro = "TVM_CONTRIB_MSC_" + StringUtils::Upper(plugin->name + "_attr") + "_H_";
    this->stack_.line("#ifndef " + macro)
        .line("#define " + macro)
        .line()
        .line("#include \"utils/plugin_utils.h\"")
        .line();
    StartNamespace();
    CodeGenAttrStruct(plugin);
    EndNamespace();
    this->stack_.line("#endif // " + macro);
  }

  /*! \brief Codegen attr struct for plugin*/
  virtual void CodeGenAttrStruct(const Plugin& plugin) {
    this->stack_.struct_start(plugin->name + "_attr").comment("define attributes");
    for (const auto& attr : plugin->attrs) {
      this->stack_.declare(attr->type, attr->name);
      if (attr->default_value.size() > 0) {
        this->stack_.declare_arg(attr->default_value);
      }
    }
    this->stack_.line()
        .comment("print method")
        .func_def("operator<<", "friend std::ostream&")
        .func_arg("out", "std::ostream&")
        .func_arg("attrs", plugin->name + "_attr&")
        .func_start()
        .line("out << \"[" + plugin->name + "_attr] : \";");
    for (const auto& attr : plugin->attrs) {
      this->stack_.line("out << \"| " + attr->name + "(" + attr->type + ")=\" << attrs." +
                        attr->name + ";");
    }
    this->stack_.func_end("out");
    CodeGenAttrSerialize(plugin);
    this->stack_.struct_end();
  }

  /*! \brief Codegen attr serilaze/deserialize for plugin*/
  virtual void CodeGenAttrSerialize(const Plugin& plugin) {}

  /*! \brief Get plugin sources*/
  virtual void CodeGenPluginSource(const Plugin& plugin) {
    CodeGenHeader(plugin);
    StartNamespace();
    CodeGenDefine(plugin);
    CodeGenRegister(plugin);
    EndNamespace();
  }

  /*! \brief Codegen define for plugin*/
  virtual void CodeGenDefine(const Plugin& plugin) {}

  /*! \brief Codegen register for plugin*/
  virtual void CodeGenRegister(const Plugin& plugin) {}

  /*! \brief Codegen cmake file*/
  virtual void CodeGenCmake(const std::set<String>& devices) {}

  /*! \brief Codegen manager member for plugin*/
  virtual void CodeGenPluginManager(const Plugin& plugin) = 0;

  /*! \brief Codegen convert function for plugin*/
  virtual void CodeGenPluginConvert(const Plugin& plugin) = 0;

  const String ToSource(const std::string& print_options = "") {
    CppPrinter printer(print_options);
    for (const auto& d : this->stack_.GetDocs()) {
      printer.Append(d);
    }
    this->stack_.Reset();
    return printer.GetString();
  };

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
