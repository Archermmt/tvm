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
 * \file src/contrib/msc/framework/tensorrt/config.h
 * \brief TensorRT config for codegen.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CONFIG_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CONFIG_H_

#include <string>

#include "../../core/codegen/base_codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief CodeGen config for tensorrt codegen
 */
struct TensorRTCodeGenConfig {
  int log_level{0};
  int profile_level{0};
  int test_iter{0};
  std::string cmake_version{"3.5"};
  std::string dataset{"Dataset"};
  std::string tensorrt_root{"/usr/local/cuda"};
  CODEGEN_CONFIG_MEMBERS
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "log_level") {
        reader->Read(&log_level);
      } else if (key == "profile_level") {
        reader->Read(&profile_level);
      } else if (key == "test_iter") {
        reader->Read(&test_iter);
      } else if (key == "cmake_version") {
        reader->Read(&cmake_version);
      } else if (key == "dataset") {
        reader->Read(&dataset);
      } else if (key == "tensorrt_root") {
        reader->Read(&tensorrt_root);
      } else {
        CODEGEN_CONFIG_PARSE
      }
    }
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CONFIG_H_
