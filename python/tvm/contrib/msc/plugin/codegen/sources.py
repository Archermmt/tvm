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

typedef enum {
  UINT8 = 0,
  INT8 = 1,
  INT16 = 2,
  INT32 = 3,
  INT64 = 4,
  FLOAT16 = 5,
  FLOAT32 = 6,
  FLOAT64 = 7,
  UNKNOWN = 8,
} MetaDataType;

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
  MetaTensor(const MetaShape& shape, const MetaDataType& data_type,
             const MetaLayout& layout = MetaLayout())
      : shape_(shape), data_type_(data_type), layout_(layout) {}

  inline void set_shape(const MetaShape& shape) { shape_ = shape; }

  inline void set_layout(const MetaLayout& layout) { layout_ = layout; }

  inline const MetaShape shape() const { return shape_; }

  inline const MetaDataType data_type() const { return data_type_; }

  inline const std::vector<int64_t> meta_shape() const { return shape_.shape(); }

  inline const MetaLayout layout() const { return layout_; }

  inline const std::string layout_name() const { return layout_.name(); }

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
  MetaDataType data_type_;
  MetaLayout layout_;
};

template <typename T>
class DataTensor : public MetaTensor {
 public:
  DataTensor(const MetaShape shape, const MetaDataType& data_type, const MetaLayout layout, T* data)
      : MetaTensor(shape, data_type, layout) {
    data_ = data;
  }

  DataTensor(const MetaShape shape, const MetaDataType& data_type, const MetaLayout layout,
             const T* data)
      : MetaTensor(shape, data_type, layout) {
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

#ifdef PLUGIN_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // PLUGIN_ENABLE_CUDA

#ifdef PLUGIN_SUPPORT_TVM
#include <tvm/relax/expr.h>

#include "tvm/../../src/contrib/msc/core/transform/layout_utils.h"
#include "tvm/../../src/contrib/msc/core/utils.h"
#ifdef PLUGIN_ENABLE_CUDA
#include "tvm/../../src/runtime/cuda/cuda_common.h"
#endif  // PLUGIN_ENABLE_CUDA
#endif  // PLUGIN_SUPPORT_TVM

#ifdef PLUGIN_SUPPORT_TORCH
#include <torch/custom_class.h>
#include <torch/script.h>
#ifdef PLUGIN_ENABLE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif  // PLUGIN_ENABLE_CUDA
#endif  // PLUGIN_SUPPORT_TORCH

#ifdef PLUGIN_SUPPORT_TENSORRT
#include "NvInfer.h"
#endif  // PLUGIN_SUPPORT_TENSORRT

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
  static std::string ToString(const std::vector<T>& value) {
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
  static void FromString(const std::string& src, std::vector<T>& target) {
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

  static void FromString(const std::string& src, std::vector<bool>& target) {
    std::vector<int> values;
    FromString(src, values);
    target.resize(values.size());
    for (size_t i = 0; i < values.size(); i++) {
      target[i] = values[i] > 0 ? true : false;
    }
  }
};

class DataUtils {
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
  static void AttrFromPrim(const PrimExpr& expr, std::string& target) {
    ICHECK(expr->IsInstance<StringImmNode>()) << "Expr is not StringImm";
    target = Downcast<StringImm>(expr)->value;
  }

  static void AttrFromPrim(const PrimExpr& expr, bool& target) {
    ICHECK(expr->IsInstance<IntImmNode>()) << "Expr is not IntImm";
    target = Downcast<IntImm>(expr)->value;
  }

  static void AttrFromPrim(const PrimExpr& expr, int& target) {
    ICHECK(expr->IsInstance<IntImmNode>()) << "Expr is not IntImm";
    target = Downcast<IntImm>(expr)->value;
  }

  static void AttrFromPrim(const PrimExpr& expr, size_t& target) {
    ICHECK(expr->IsInstance<IntImmNode>()) << "Expr is not IntImm";
    target = Downcast<IntImm>(expr)->value;
  }

  static void AttrFromPrim(const PrimExpr& expr, long& target) {
    ICHECK(expr->IsInstance<IntImmNode>()) << "Expr is not IntImm";
    target = Downcast<IntImm>(expr)->value;
  }

  static void AttrFromPrim(const PrimExpr& expr, long long& target) {
    ICHECK(expr->IsInstance<IntImmNode>()) << "Expr is not IntImm";
    target = Downcast<IntImm>(expr)->value;
  }

  static void AttrFromPrim(const PrimExpr& expr, float& target) {
    ICHECK(expr->IsInstance<FloatImmNode>()) << "Expr is not FloatImm";
    target = Downcast<FloatImm>(expr)->value;
  }

  static void AttrFromPrim(const PrimExpr& expr, double& target) {
    ICHECK(expr->IsInstance<FloatImmNode>()) << "Expr is not FloatImm";
    target = Downcast<FloatImm>(expr)->value;
  }

  template <typename T>
  static void AttrFromPrims(const Tuple& tuple, std::vector<T>& target) {
    for (size_t i = 0; i < tuple->fields.size(); i++) {
      ICHECK(tuple->fields[i]->IsInstance<PrimExprNode>()) << "Field is not PrimExpr";
      AttrFromPrim(tuple->fields[i], target[i]);
    }
  }

  static void AttrFromArg(const TVMArgValue& arg, std::string& target) {
    target = arg.operator std::string();
  }

  static void AttrFromArg(const TVMArgValue& arg, bool& target) { target = arg; }

  static void AttrFromArg(const TVMArgValue& arg, int& target) { target = arg; }

  static void AttrFromArg(const TVMArgValue& arg, size_t& target) { target = int(arg); }

  static void AttrFromArg(const TVMArgValue& arg, long& target) { target = int64_t(arg); }

  static void AttrFromArg(const TVMArgValue& arg, long long& target) { target = int64_t(arg); }

  static void AttrFromArg(const TVMArgValue& arg, float& target) { target = double(arg); }

  static void AttrFromArg(const TVMArgValue& arg, double& target) { target = arg; }

  template <typename T>
  static void AttrFromArgs(const TVMArgs& args, size_t start, size_t num, std::vector<T>& target) {
    for (size_t i = 0; i < num; i++) {
      AttrFromArg(args[start + i], target[i]);
    }
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

  static MetaShape ToMetaShape(const Optional<Array<PrimExpr>>& tvm_shape) {
    if (tvm_shape.defined()) {
      std::vector<int64_t> shape_data;
      for (auto s : tvm_shape.value()) {
        if (s->IsInstance<tvm::IntImmNode>()) {
          shape_data.push_back(Downcast<Integer>(s)->value);
        } else {
          shape_data.push_back(-1);
        }
      }
      return MetaShape(shape_data);
    }
    return MetaShape();
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

  static MetaTensor ToMetaTensor(const Expr& expr,
                                 const LayoutDecision& layout_dec = LayoutDecision()) {
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(expr);
    if (layout_dec.defined() && layout_dec->layout.defined()) {
      const auto& layout = MetaLayout(layout_dec->layout.name());
      return MetaTensor(ToMetaShape(sinfo->GetShape()), ToMetaType(sinfo->dtype), layout);
    }
    const auto& layout = MetaLayout(SpanUtils::GetAttr(expr->span, "layout"));
    return MetaTensor(ToMetaShape(sinfo->GetShape()), ToMetaType(sinfo->dtype), layout);
  }

  template <typename T>
  static DataTensor<T> ToDataTensor(DLTensor* tensor, bool read_only) {
    if (read_only) {
      return DataTensor<T>(ToMetaShape(tensor, false), ToMetaType(tensor->dtype), MetaLayout(),
                           (const T*)(tensor->data));
    } else {
      return DataTensor<T>(ToMetaShape(tensor, false), ToMetaType(tensor->dtype), MetaLayout(),
                           (T*)(tensor->data));
    }
  }

  static DataType ToTVMType(const MetaDataType& dtype) {
    DataType tvm_type;
    if (dtype == MetaDataType::INT8) {
      tvm_type = DataType::Int(8);
    } else if (dtype == MetaDataType::INT16) {
      tvm_type = DataType::Int(16);
    } else if (dtype == MetaDataType::INT32) {
      tvm_type = DataType::Int(32);
    } else if (dtype == MetaDataType::INT64) {
      tvm_type = DataType::Int(64);
    } else if (dtype == MetaDataType::FLOAT16) {
      tvm_type = DataType::Float(16);
    } else if (dtype == MetaDataType::FLOAT32) {
      tvm_type = DataType::Float(32);
    } else if (dtype == MetaDataType::FLOAT64) {
      tvm_type = DataType::Float(64);
    } else {
      throw std::runtime_error("Unsupported type");
    }
    return tvm_type;
  }

  static DataType ToTVMType(const std::string& dtype) {
    return ToTVMType(DataUtils::ToMetaType(dtype));
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

  static void FillDLShape(const MetaShape& shape, DLTensor* data) {
    auto shape_data = static_cast<int64_t*>(data->data);
    for (size_t i = 0; i < shape.Ndim(); i++) {
      shape_data[i] = shape.DimAt(i);
    }
  }

  static TensorStructInfo ToTensorStructInfo(const MetaTensor& tensor,
                                             const Optional<VDevice>& device) {
    const auto& t_shape = ToTVMShape(tensor.shape());
    const auto& t_type = ToTVMType(tensor.data_type());
    return TensorStructInfo(ShapeExpr(t_shape), t_type, device);
  }

  static TensorStructInfo ToTensorStructInfo(const MetaTensor& tensor, const Expr& expr) {
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(expr);
    return ToTensorStructInfo(tensor, sinfo->vdevice);
  }

  static bool OnDevice(DLTensor* tensor, DLDeviceType device) {
    return tensor->device.device_type == device;
  }

  static void CheckDevice(DLTensor* tensor, DLDeviceType device) {
    ICHECK_EQ(tensor->device.device_type, device);
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
  static MetaDataType ToMetaType(const torch::ScalarType& dtype) {
    MetaDataType meta_type;
    if (dtype == torch::kChar) {
      meta_type = MetaDataType::INT8;
    } else if (dtype == torch::kInt) {
      meta_type = MetaDataType::INT32;
    } else if (dtype == torch::kInt64) {
      meta_type = MetaDataType::INT64;
    } else if (dtype == torch::kLong) {
      meta_type = MetaDataType::INT64;
    } else if (dtype == torch::kFloat16) {
      meta_type = MetaDataType::FLOAT16;
    } else if (dtype == torch::kFloat) {
      meta_type = MetaDataType::FLOAT32;
    } else if (dtype == torch::kDouble) {
      meta_type = MetaDataType::FLOAT64;
    } else {
      meta_type = MetaDataType::UNKNOWN;
    }
    return meta_type;
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
    return MetaTensor(ToMetaShape(tensor), ToMetaType(tensor.scalar_type()), layout);
  }

  template <typename T>
  static DataTensor<T> ToDataTensor(const torch::Tensor& tensor, const MetaTensor& meta,
                                    bool read_only) {
    if (read_only) {
      return DataTensor<T>(meta.shape(), meta.data_type(), meta.layout(),
                           (const T*)(tensor.data_ptr()));
    } else {
      return DataTensor<T>(meta.shape(), meta.data_type(), meta.layout(), (T*)(tensor.data_ptr()));
    }
  }

  static torch::ScalarType ToTorchType(const MetaDataType& dtype) {
    torch::ScalarType torch_type;
    if (dtype == MetaDataType::INT8) {
      torch_type = torch::kChar;
    } else if (dtype == MetaDataType::INT32) {
      torch_type = torch::kInt;
    } else if (dtype == MetaDataType::INT64) {
      torch_type = torch::kInt64;
    } else if (dtype == MetaDataType::FLOAT16) {
      torch_type = torch::kFloat16;
    } else if (dtype == MetaDataType::FLOAT32) {
      torch_type = torch::kFloat;
    } else if (dtype == MetaDataType::FLOAT64) {
      torch_type = torch::kDouble;
    } else {
      throw std::runtime_error("Unsupported type");
    }
    return torch_type;
  }

  static torch::ScalarType ToTorchType(const std::string& dtype) {
    return ToTorchType(DataUtils::ToMetaType(dtype));
  }

  static torch::Device ToTorchDevice(const std::string& device) {
    if (device == "cpu") {
      return torch::Device(torch::kCPU);
    }
    if (device == "cuda") {
      return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
  }

  static torch::Tensor MallocTorchTensor(const MetaTensor& tensor, const torch::Device& device) {
    auto t_type = ToTorchType(tensor.data_type());
    auto opt = torch::TensorOptions().dtype(t_type).device(device);
    return torch::zeros(tensor.meta_shape(), opt);
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
