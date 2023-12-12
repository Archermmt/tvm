# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""tvm.contrib.msc.plugin.codegen.sources"""

from typing import Dict


def get_plugin_base_h_code() -> str:
    """Create plugin base header file codes

    Returns
    -------
    source: str
        The plugin base header source.
    """

    return """#ifndef TVM_CONTRIB_MSC_UTILS_PLUGIN_BASE_H_
#define TVM_CONTRIB_MSC_UTILS_PLUGIN_BASE_H_

#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace tvm {
namespace contrib {
namespace msc {

enum MetaDataType {
  UINT8 = 0,
  INT8 = 1,
  INT16 = 2,
  INT32 = 3,
  INT64 = 4,
  FLOAT16 = 5,
  FLOAT32 = 6,
  FLOAT64 = 7,
  UNKNOWN
};

class MetaShape {
 public:
  MetaShape() { shape_.resize(0); }

  MetaShape(const std::vector<int64_t>& shape) {
    for (auto d : shape) {
      shape_.push_back(d);
    }
  }

  template <typename T>
  void SetShape(const std::vector<T>& shape) {
    for (auto d : shape) {
      shape_.push_back(static_cast<int64_t>(d));
    }
  }

  template <typename T>
  void SetDim(int index, T dim) {
    int valid_index = index < 0 ? shape_.size() + index : index;
    if (valid_index >= shape_.size()) {
      std::string err =
          std::to_string(index) + " out of dims size " + std::to_string(shape_.size());
      throw std::runtime_error(err);
    }
    shape_[valid_index] = dim;
  }

  template <typename T>
  const std::vector<T> GetShape() const {
    std::vector<T> shape;
    for (auto d : shape_) {
      shape.push_back(d);
    }
    return shape;
  }

  inline int64_t DimAt(int index) const {
    int valid_index = index < 0 ? shape_.size() + index : index;
    if (valid_index >= shape_.size()) {
      std::string err =
          std::to_string(index) + " out of dims size " + std::to_string(shape_.size());
      throw std::runtime_error(err);
    }
    return shape_[valid_index];
  }

  inline size_t Ndim() const { return shape_.size(); }

  inline const std::vector<int64_t> shape() const { return shape_; }

  inline size_t size() const {
    size_t size = 1;
    for (auto d : shape_) {
      assert(d > 0 && "Can not compute static size with unknow dim");
      size *= d;
    }
    return size;
  }

  inline int64_t operator[](int index) const { return DimAt(index); }

  friend std::ostream& operator<<(std::ostream& out, const MetaShape& shape) {
    for (size_t i = 0; i < shape.Ndim(); i++) {
      out << shape.DimAt(i) << (1 < shape.Ndim() ? "" : ",");
    }
    return out;
  }

 private:
  std::vector<int64_t> shape_;
};

class MetaLayoutAxis {
 public:
  MetaLayoutAxis(const char name, size_t factor = 0) : factor_(factor) {
    name_ = (factor == 0 ? "" : std::to_string(factor)) + std::string(1, name);
  }

  MetaLayoutAxis(const std::string& name) {
    if (name.size() == 1) {
      factor_ = 0;
      name_ = name;
    } else {
      factor_ = std::stoi(name.substr(1));
      name_ = name.substr(0, 1);
    }
  }

  inline const std::string name() const { return name_; }

  inline size_t factor() const { return factor_; }

 private:
  std::string name_;
  size_t factor_;
};

class MetaLayout {
 public:
  MetaLayout() {}

  MetaLayout(const std::string& name) : name_(name) {
    int factor = 0;
    for (char c : name) {
      if (c >= 'A' && c <= 'Z') {
        assert(factor == 0 && "Upper layout axis do not accept factor");
        MetaLayoutAxis axis(c);
        axes_.push_back(axis);
      } else if (c >= 'a' && c <= 'z') {
        assert(factor > 0 && "Lower layout axis should has factor");
        MetaLayoutAxis axis(c, factor);
        axes_.push_back(axis);
        factor = 0;
      } else if (c >= '0' && c <= '9') {
        assert(factor >= 0 && "Factor number should between 0 and 9");
        factor = factor * 10 + c - '0';
      } else {
        throw std::runtime_error("Unexpected layout axis " + name);
      }
    }
    CheckValid();
  }

  MetaLayout(const std::vector<MetaLayoutAxis>& axes) : axes_(axes) {
    name_ = "";
    for (auto a : axes_) {
      name_ += (a.factor() == 0 ? "" : std::to_string(a.factor())) + a.name();
    }
    CheckValid();
  };

  void CheckValid() {
    std::set<std::string> recorded_axes;
    for (auto a : axes_) {
      auto axis_name = a.name();
      assert(!recorded_axes.count(axis_name) && ("Has duplicate layout axis in " + name_).c_str());
      recorded_axes.insert(axis_name);
    }
  }

  inline const MetaLayoutAxis AxisAt(int index) const {
    int valid_index = index < 0 ? axes_.size() + index : index;
    if (valid_index >= axes_.size()) {
      std::string err = std::to_string(index) + " out of axes size " + std::to_string(axes_.size());
      throw std::runtime_error(err);
    }
    return axes_[valid_index];
  }

  inline MetaLayoutAxis operator[](int index) { return AxisAt(index); }

  inline size_t ndim() const { return axes_.size(); }

  inline std::string name() const { return name_; }

  friend std::ostream& operator<<(std::ostream& out, const MetaLayout& layout) {
    out << layout.name();
    return out;
  }

 private:
  std::string name_;
  std::vector<MetaLayoutAxis> axes_;
};

class MetaTensor {
 public:
  MetaTensor(const MetaShape& shape, const MetaLayout& layout = MetaLayout())
      : shape_(shape), layout_(layout) {}

  inline void set_shape(const MetaShape& shape) { shape_ = shape; }

  inline void set_layout(const MetaLayout& layout) { layout_ = layout; }

  inline const MetaShape shape() const { return shape_; }

  inline const std::vector<int64_t> meta_shape() const { return shape_.shape(); }

  inline const MetaLayout layout() const { return layout_; }

  inline size_t Ndim() const { return shape_.Ndim(); }

  inline size_t size() const { return shape_.size(); }

  inline int64_t DimAt(int index) const { return shape_.DimAt(index); }

  inline MetaLayoutAxis AxisAt(int index) const { return layout_.AxisAt(index); }

  int64_t dim_at(const std::string& axis_name) const {
    for (size_t i = 0; i < layout_.ndim(); i++) {
      if (layout_.AxisAt(i).name() == axis_name) {
        return shape_.DimAt(i);
      }
    }
    throw std::runtime_error("Can not find dim for " + axis_name);
  }

  int64_t DimAt(const MetaLayoutAxis& axis) const { return DimAt(axis.name()); }

  friend std::ostream& operator<<(std::ostream& out, const MetaTensor& tensor) {
    out << "tensor : <" << tensor.shape() << ">, (" << tensor.layout() << ")";
    return out;
  }

 private:
  MetaShape shape_;
  MetaLayout layout_;
};

template <typename T>
class DataTensor : public MetaTensor {
 public:
  DataTensor(const MetaShape shape, const MetaLayout layout, T* data) : MetaTensor(shape, layout) {
    data_ = data;
  }

  DataTensor(const MetaShape shape, const MetaLayout layout, const T* data)
      : MetaTensor(shape, layout) {
    data_ = const_cast<T*>(data);
  }

  T* data() const { return data_; }

  const T* const_data() const { return data_; }

 private:
  T* data_{nullptr};
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_UTILS_PLUGIN_BASE_H_
"""


def get_plugin_utils_h_code() -> str:
    """Create plugin utils header file codes

    Returns
    -------
    source: str
        The plugin utils header source.
    """

    return """#ifndef TVM_CONTRIB_MSC_UTILS_PLUGIN_UTILS_H_
#define TVM_CONTRIB_MSC_UTILS_PLUGIN_UTILS_H_

#include <stdio.h>
#include <string.h>

#include <cassert>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#include "plugin_base.h"

#ifdef PLUGIN_SUPPORT_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef PLUGIN_SUPPORT_TVM
#include <tvm/relax/expr.h>
#include <tvm/runtime/vm/memory_manager.h>
#endif

#ifdef PLUGIN_SUPPORT_TORCH
#include <torch/script.h>
#endif

#ifdef PLUGIN_SUPPORT_TENSORRT
#include "NvInfer.h"
#endif

namespace tvm {
namespace contrib {
namespace msc {

class SerializeUtils {
 public:
  // Helper function for serializing plugin attrs
  template <typename T>
  static void WriteBuffer(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }

  static void WriteBuffer(char*& buffer, const std::string& val) {
    *reinterpret_cast<size_t*>(buffer) = val.size();
    buffer += sizeof(size_t);
    val.copy(buffer, val.size());
    buffer += sizeof(char) * val.size();
  }

  template <typename T>
  static void WriteBuffer(char*& buffer, const std::vector<T>& val) {
    *reinterpret_cast<size_t*>(buffer) = val.size();
    buffer += sizeof(size_t);
    for (auto e : val) {
      SerializeUtils::WriteBuffer(buffer, e);
    }
  }

  // Helper function for deserializing plugin attrs
  template <typename T>
  static void ReadBuffer(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
  }

  static void ReadBuffer(const char*& buffer, std::string& val) {
    auto size = *reinterpret_cast<const size_t*>(buffer);
    buffer += sizeof(size_t);
    val = std::string(reinterpret_cast<const char*>(buffer), size);
    buffer += sizeof(char) * size;
  }

  template <typename T>
  static void ReadBuffer(const char*& buffer, std::vector<T>& val) {
    auto size = *reinterpret_cast<const size_t*>(buffer);
    buffer += sizeof(size_t);
    val.resize(size);
    for (size_t i = 0; i < size; i++) {
      ReadBuffer(buffer, val[i]);
    }
  }

  template <typename T>
  static const std::string ToString(const T& value) {
    return std::to_string(value);
  }

  static std::string ToString(const std::string& value) { return value; }

  template <typename T>
  static std::string VecToString(const std::vector<T>& value) {
    std::string str = std::to_string(value.size());
    for (const auto& v : value) {
      str += "," + std::to_string(v);
    }
    return str;
  }

  static void FromString(const std::string& src, std::string& target) { target = src; }

  static void FromString(const std::string& src, bool& target) {
    target = std::stoi(src) > 0 ? true : false;
  }

  static void FromString(const std::string& src, int& target) { target = std::stoi(src); }

  static void FromString(const std::string& src, size_t& target) { target = std::stoi(src); }

  static void FromString(const std::string& src, long& target) { target = std::stol(src); }

  static void FromString(const std::string& src, long long& target) { target = std::stoll(src); }

  static void FromString(const std::string& src, float& target) { target = std::stod(src); }

  static void FromString(const std::string& src, double& target) { target = std::stof(src); }

  template <typename T>
  static void VecFromString(const std::string& src, std::vector<T>& target) {
    std::string left_str = src;
    int pos = left_str.find(",");
    if (pos == std::string::npos) {
      return;
    }
    assert(pos > 0);
    size_t src_size;
    FromString(left_str.substr(0, pos), src_size);
    target.resize(src_size);
    for (size_t i = 0; i < src_size; i++) {
      pos = left_str.find(",");
      left_str = left_str.substr(pos + 1);
      FromString(left_str, target[i]);
    }
  }

  static void VecFromString(const std::string& src, std::vector<bool>& target) {
    std::vector<int> values;
    VecFromString(src, values);
    target.resize(values.size());
    for (size_t i = 0; i < values.size(); i++) {
      target[i] = values[i] > 0 ? true : false;
    }
  }
};

class DataTypeUtils {
 public:
  static MetaDataType ToMetaType(const std::string& name) {
    MetaDataType dtype;
    if (name == "int8") {
      dtype = MetaDataType::INT8;
    } else if (name == "uint8" || name == "char") {
      dtype = MetaDataType::UINT8;
    } else if (name == "int16") {
      dtype = MetaDataType::INT16;
    } else if (name == "int32" || name == "int") {
      dtype = MetaDataType::INT32;
    } else if (name == "int64" || name == "long" || name == "long long") {
      dtype = MetaDataType::INT64;
    } else if (name == "float16" || name == "half") {
      dtype = MetaDataType::FLOAT16;
    } else if (name == "float32" || name == "float") {
      dtype = MetaDataType::FLOAT32;
    } else if (name == "float64" || name == "double") {
      dtype = MetaDataType::FLOAT64;
    } else {
      dtype = MetaDataType::UNKNOWN;
    }
    return dtype;
  }
};

#ifdef PLUGIN_SUPPORT_TVM
using namespace tvm::relax;
using namespace tvm::runtime;
class TVMUtils {
 public:
  static DataType ToTVMType(const std::string dtype) {
    DataType tvm_type;
    if (dtype == "int8") {
      tvm_type = DataType::Int(8);
    } else if (dtype == "uint8" || dtype == "char") {
      tvm_type = DataType::UInt(8);
    } else if (dtype == "int16") {
      tvm_type = DataType::Int(16);
    } else if (dtype == "int32" || dtype == "int") {
      tvm_type = DataType::Int(32);
    } else if (dtype == "int64" || dtype == "long" || dtype == "long long") {
      tvm_type = DataType::Int(64);
    } else if (dtype == "float16" || dtype == "half") {
      tvm_type = DataType::Float(16);
    } else if (dtype == "float32" || dtype == "float") {
      tvm_type = DataType::Float(32);
    } else if (dtype == "float64" || dtype == "double") {
      tvm_type = DataType::Float(64);
    } else {
      throw std::runtime_error(("Unsupported type " + dtype).c_str());
    }
    return tvm_type;
  }

  static MetaDataType ToMetaType(const DataType& dtype) {
    MetaDataType meta_type;
    if (dtype.code() == 0 && dtype.bits() == 8) {
      meta_type = MetaDataType::INT8;
    } else if (dtype.code() == 0 && dtype.bits() == 16) {
      meta_type = MetaDataType::INT16;
    } else if (dtype.code() == 0 && dtype.bits() == 32) {
      meta_type = MetaDataType::INT32;
    } else if (dtype.code() == 0 && dtype.bits() == 64) {
      meta_type = MetaDataType::INT64;
    } else if (dtype.code() == 1 && dtype.bits() == 8) {
      meta_type = MetaDataType::UINT8;
    } else if (dtype.code() == 2 && dtype.bits() == 16) {
      meta_type = MetaDataType::FLOAT16;
    } else if (dtype.code() == 2 && dtype.bits() == 32) {
      meta_type = MetaDataType::FLOAT32;
    } else if (dtype.code() == 2 && dtype.bits() == 64) {
      meta_type = MetaDataType::FLOAT64;
    } else {
      meta_type = MetaDataType::UNKNOWN;
    }
    return meta_type;
  }

  static MetaDataType ToMetaType(const DLDataType& dtype) {
    MetaDataType meta_type;
    if (dtype.code == 0U && dtype.bits == 8) {
      meta_type = MetaDataType::INT8;
    } else if (dtype.code == 0U && dtype.bits == 16) {
      meta_type = MetaDataType::INT16;
    } else if (dtype.code == 0U && dtype.bits == 32) {
      meta_type = MetaDataType::INT32;
    } else if (dtype.code == 0U && dtype.bits == 64) {
      meta_type = MetaDataType::INT64;
    } else if (dtype.code == 1U && dtype.bits == 8) {
      meta_type = MetaDataType::UINT8;
    } else if (dtype.code == 2U && dtype.bits == 16) {
      meta_type = MetaDataType::FLOAT16;
    } else if (dtype.code == 2U && dtype.bits == 32) {
      meta_type = MetaDataType::FLOAT32;
    } else if (dtype.code == 2U && dtype.bits == 64) {
      meta_type = MetaDataType::FLOAT64;
    } else {
      meta_type = MetaDataType::UNKNOWN;
    }
    return meta_type;
  }

  static Array<tvm::PrimExpr> ToTVMShape(const MetaShape& meta_shape) {
    Array<tvm::PrimExpr> tvm_shape;
    for (size_t i = 0; i < meta_shape.Ndim(); i++) {
      auto dim = meta_shape.DimAt(i);
      if (dim == -1) {
        tvm_shape.push_back(tir::Any());
      } else {
        tvm_shape.push_back(Integer(dim));
      }
    }
    return tvm_shape;
  }

  static MetaShape ToMetaShape(const Array<tvm::PrimExpr>& tvm_shape) {
    std::vector<int64_t> shape_data;
    for (auto s : tvm_shape) {
      if (s->IsInstance<tvm::IntImmNode>()) {
        shape_data.push_back(Downcast<Integer>(s)->value);
      } else {
        shape_data.push_back(-1);
      }
    }
    return MetaShape(shape_data);
  }

  static MetaShape ToMetaShape(DLTensor* tensor, bool as_data = true) {
    std::vector<int64_t> dims;
    if (as_data) {
      assert(tensor->ndim == 1);
      assert(TVMUtils::ToMetaType(tensor->dtype) == MetaDataType::INT64);
      int64_t* data_ptr = (int64_t*)tensor->data;
      for (size_t i = 0; i < tensor->shape[0]; i++) {
        dims.push_back(data_ptr[i]);
      }
    } else {
      for (size_t i = 0; i < tensor->ndim; i++) {
        dims.push_back(tensor->shape[i]);
      }
    }
    return MetaShape(dims);
  }

  static void FillDLShape(const MetaShape& shape, DLTensor* data) {
    auto shape_data = static_cast<int64_t*>(data->data);
    for (size_t i = 0; i < shape.Ndim(); i++) {
      shape_data[i] = shape.DimAt(i);
    }
  }

  template <typename T>
  static DataTensor<T> ToDataTensor(DLTensor* tensor, bool read_only,
                                    const std::string& layout = "") {
    if (read_only) {
      return DataTensor<T>(TVMUtils::ToMetaShape(tensor, false), MetaLayout(layout),
                           (const T*)(tensor->data));
    } else {
      return DataTensor<T>(TVMUtils::ToMetaShape(tensor, false), MetaLayout(layout),
                           (T*)(tensor->data));
    }
  }

  static void CheckDevice(DLTensor* tensor, DLDeviceType device) {
    ICHECK_EQ(tensor->device.device_typ, device);
  }

  static Device DefaultCPU() {
    Device cpu_dev{kDLCPU, 0};
    return cpu_dev;
  }

  static Device DefaultCUDA() {
    Device cuda_dev{kDLCUDA, 0};
    return cuda_dev;
  }
};
#endif  // PLUGIN_SUPPORT_TVM

#ifdef PLUGIN_SUPPORT_TORCH
class TorchUtils {
 public:
  static torch::ScalarType ToTorchType(const std::string& dtype) {
    torch::ScalarType torch_type;
    if (dtype == "int8") {
      torch_type = torch::kChar;
    } else if (dtype == "uint8") {
      torch_type = torch::kUInt8;
    } else if (dtype == "char") {
      torch_type = torch::kChar;
    } else if (dtype == "int32" || dtype == "int") {
      torch_type = torch::kInt;
    } else if (dtype == "int64") {
      torch_type = torch::kInt64;
    } else if (dtype == "long" || dtype == "long long") {
      torch_type = torch::kLong;
    } else if (dtype == "float16" || dtype == "half") {
      torch_type = torch::kFloat16;
    } else if (dtype == "float32" || dtype == "float") {
      torch_type = torch::kFloat;
    } else if (dtype == "float64" || dtype == "double") {
      torch_type = torch::kDouble;
    } else {
      throw std::runtime_error(("Unsupported type " + dtype).c_str());
    }
    return torch_type;
  }

  static std::string TypeName(const torch::ScalarType& dtype) {
    std::string dtype_name;
    if (dtype == torch::kChar) {
      dtype_name = "char";
    } else if (dtype == torch::kInt) {
      dtype_name = "int";
    } else if (dtype == torch::kInt64) {
      dtype_name = "int64";
    } else if (dtype == torch::kLong) {
      dtype_name = "long";
    } else if (dtype == torch::kFloat16) {
      dtype_name = "float16";
    } else if (dtype == torch::kFloat) {
      dtype_name = "float32";
    } else if (dtype == torch::kDouble) {
      dtype_name = "double";
    } else {
      dtype_name = "unknown";
    }
    return dtype_name;
  }

  static MetaShape ToMetaShape(const torch::Tensor& tensor) {
    std::vector<int64_t> shape_data;
    for (size_t idx = 0; idx < tensor.dim(); idx++) {
      shape_data.push_back(tensor.size(idx));
    }
    return MetaShape(shape_data);
  }

  static MetaTensor ToMetaTensor(const torch::Tensor& tensor,
                                 const MetaLayout& layout = MetaLayout()) {
    return MetaTensor(TorchUtils::ToMetaShape(tensor), layout);
  }

  template <typename T>
  static DataTensor<T> ToDataTensor(const torch::Tensor& tensor, const MetaLayout& layout,
                                    bool read_only) {
    if (read_only) {
      return DataTensor<T>(TorchUtils::ToMetaShape(tensor), layout, (const T*)(tensor.data_ptr()));
    } else {
      return DataTensor<T>(TorchUtils::ToMetaShape(tensor), layout, (T*)(tensor.data_ptr()));
    }
  }

  template <typename T>
  static std::vector<DataTensor<T>> ToDataTensors(const std::vector<torch::Tensor>& tensors,
                                                  const std::vector<MetaLayout>& layouts,
                                                  bool read_only) {
    std::vector<DataTensor<T>> data_tensors;
    for (size_t i = 0; i < tensors.size(); i++) {
      data_tensors.push_back(TorchUtils::ToDataTensor<T>(tensors[i], layouts[i], read_only));
    }
    return data_tensors;
  }
};
#endif  // PLUGIN_SUPPORT_TORCH

#ifdef PLUGIN_SUPPORT_TENSORRT

#ifndef TRT_VERSION_GE
#define TRT_VERSION_GE(major, minor, patch)                            \\
  ((TRT_MAJOR > major) || (TRT_MAJOR == major && TRT_MINOR > minor) || \\
   (TRT_MAJOR == major && TRT_MINOR == minor && TRT_PATCH >= patch))
#endif

using namespace nvinfer1;

class TRTUtils {
 public:
  static MetaShape ToMetaShape(const Dims& trt_dims, bool use_implicit_batch = true) {
    std::vector<int64_t> dims;
    if (use_implicit_batch) {
      dims.push_back(1);
    }
    auto trt_dims = trt_desc.desc.dims;
    for (size_t idx = 0; idx < trt_dims.nbDims; idx++) {
      dims.push_back(trt_dims.d[idx]);
    }
    return MetaShape(dims);
  }

  static MetaShape ToMetaShape(const DynamicPluginTensorDesc& trt_desc,
                               bool use_implicit_batch = true) {
    std::vector<int64_t> dims;
    if (use_implicit_batch) {
      dims.push_back(1);
    }
    auto trt_dims = trt_desc.desc.dims;
    for (size_t idx = 0; idx < trt_dims.nbDims; idx++) {
      dims.push_back(trt_dims.d[idx]);
    }
    return MetaShape(dims);
  }

  static MetaShape ToMetaShape(const DimsExprs& trt_dims, bool use_implicit_batch = true) {
    std::vector<int64_t> dims;
    if (use_implicit_batch) {
      dims.push_back(1);
    }
    for (size_t idx = 0; idx < trt_dims.nbDims; idx++) {
      assert(trt_dims.d[idx]->isConstant());
      dims.push_back(trt_dims.d[idx]->getConstantValue());
    }
    return MetaShape(dims);
  }

  static Dims ToDims(const MetaShape& meta_shape, bool use_implicit_batch = true) {
    std::vector<int64_t> int_dims;
    if (use_implicit_batch) {
      for (size_t i = 1; i < meta_shape.Ndim(); i++) {
        int_dims.push_back(meta_shape.DimAt(i));
      }
    } else {
      for (size_t i = 0; i < meta_shape.Ndim(); i++) {
        int_dims.push_back(meta_shape.DimAt(i));
      }
    }
    Dims dims{int(int_dims.size())};
    for (size_t i = 0; i < int_dims.size(); i++) {
      dims.d[i] = int_dims[i];
    }
    return dims;
  }

  static DimsExprs ToDimsExprs(const MetaShape& meta_shape, IExprBuilder& exprBuilder,
                               bool use_implicit_batch = true) {
    std::vector<int64_t> int_dims;
    if (use_implicit_batch) {
      for (size_t i = 1; i < meta_shape.Ndim(); i++) {
        int_dims.push_back(meta_shape.DimAt(i));
      }
    } else {
      for (size_t i = 0; i < meta_shape.Ndim(); i++) {
        int_dims.push_back(meta_shape.DimAt(i));
      }
    }
    DimsExprs dims{int(int_dims.size())};
    for (size_t i = 0; i < int_dims.size(); i++) {
      dims.d[i] = exprBuilder.constant(int_dims[i]);
    }
    return dims;
  }
};
#endif  // PLUGIN_SUPPORT_TENSORRT

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_UTILS_PLUGIN_UTILS_H_
"""


def get_plugin_sources() -> Dict[str, str]:
    """Create base sources for plugin codegen

    Returns
    -------
    sources: dict<str,str>
        The base utils sources.
    """

    return {"plugin_base.h": get_plugin_base_h_code(), "plugin_utils.h": get_plugin_utils_h_code()}
