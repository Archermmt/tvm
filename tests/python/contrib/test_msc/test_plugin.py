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

""" Test Plugin in MSC. """

import pytest

import torch
from torch import nn

import tvm.testing
from tvm.contrib.msc.plugin import build_plugins_manager
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils

requires_tensorrt = pytest.mark.skipif(
    tvm.get_global_func("relax.ext.tensorrt", True) is None,
    reason="TENSORRT is not enabled",
)


def _get_externs_header():
    """Get the header source for externs"""

    return """#ifndef EXTERNS_H_
#define EXTERNS_H_

#include "utils/plugin_utils.h"

#ifdef PLUGIN_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tvm {
namespace contrib {
namespace msc {

template <typename TAttr>
std::vector<MetaTensor> my_relu_infer(const std::vector<MetaTensor>& inputs, const TAttr& attrs,
                                      bool is_runtime) {
  std::vector<MetaTensor> outputs;
  outputs.push_back(MetaTensor(inputs[0].shape(), inputs[0].layout()));
  return outputs;
}

template <typename T>
void my_relu_cpu_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val);

template <typename T, typename TAttr>
void my_relu_cpu_compute(const DataTensor<T>& input, DataTensor<T>& output, const TAttr& attrs) {
  my_relu_cpu_kernel(input, output, T(attrs.max_val));
}

#ifdef PLUGIN_ENABLE_CUDA
template <typename T>
void my_relu_cuda_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val,
                         const rtStream_t& stream);

template <typename T, typename TAttr>
void my_relu_cuda_compute(const DataTensor<T>& input, DataTensor<T>& output, const TAttr& attrs,
                          const rtStream_t& stream) {
  my_relu_cuda_kernel(input, output, T(attrs.max_val), stream);
}
#endif

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // EXTERNS_H_
"""


def _get_externs_cc():
    """Get externs cc source"""
    return """#include "externs.h"

namespace tvm {
namespace contrib {
namespace msc {

template <typename T>
void my_relu_cpu_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val) {
  const T* input_data = input.const_data();
  T* output_data = output.data();
  for (size_t i = 0; i < output.size(); i++) {
    if (input_data[i] >= max_val) {
      output_data[i] = max_val;
    } else if (input_data[i] <= 0) {
      output_data[i] = 0;
    } else {
      output_data[i] = input_data[i];
    }
  }
}

template void my_relu_cpu_kernel<float>(const DataTensor<float>& input, DataTensor<float>& output,
                                        float max_val);
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
"""


def _get_externs_cu():
    """Get externs cu source"""

    return """#include "externs.h"

#define CU1DBLOCK 256
#define KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDims.x * gridDim.x)

namespace tvm {
namespace contrib {
namespace msc {

inline int n_blocks(int size, int block_size) {
  return size / block_size + (size % block_size == 0 ? 0 : 1);
}

template <typename T>
__global__ static void _my_relu(const T* src, const T* dst, T max_val, int n) {
  KERNEL_LOOP(i, n) {
    if (src[i] >= max_val) {
      dst[i] = max_val;
    } else if (src[i] <= 0) {
      dst[i] = 0;
    } else {
      dst[i] = src[i];
    }
  }
}

template <typename T>
void my_relu_cuda_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val,
                         const rtStream_t& stream) {
  const T* input_data = input.const_data();
  T* output_data = output.data();
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(output.static_size(), CU1DBLOCK));
  _my_relu<<<Gr, Bl, 0, stream>>>(input_data, output_data, max_val, output.static_size());
}

template void my_relu_cuda_kernel<float>(const DataTensor<float>& input, DataTensor<float>& output,
                                         float max_val, const rtStream_t& stream);
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
"""


def _create_plugin(externs_dir):
    """Create sources under source folder"""
    with open(externs_dir.relpath("externs.h"), "w") as f:
        f.write(_get_externs_header())
    with open(externs_dir.relpath("externs.cc"), "w") as f:
        f.write(_get_externs_cc())
    with open(externs_dir.relpath("externs.cu"), "w") as f:
        f.write(_get_externs_cu())
    return {
        "my_relu": {
            "inputs": [{"name": "input", "dtype": "T"}],
            "outputs": [{"name": "output", "dtype": "T"}],
            "attrs": [{"name": "max_val", "type": "float"}],
            "support_dtypes": {"T": ["float"]},
            "externs": {
                "infer_output": {"name": "my_relu_infer", "header": "externs.h"},
                "cpu_compute": {
                    "name": "my_relu_cpu_compute",
                    "header": "externs.h",
                    "source": "externs.cc",
                },
                "cuda_compute": {
                    "name": "my_relu_cuda_compute",
                    "header": "externs.h",
                    "source": "externs.cu",
                },
            },
        }
    }


def _get_model(torch_manager):
    """Build model with plugin"""

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)
            self.relu = torch_manager.my_relu(max_val=12)
            self.maxpool = nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, c):
            x = self.conv(x)
            x = self.relu(x)
            return self.maxpool(x)

    return MyModel()


def test_torch_plugin():
    """Test plugin in torch"""

    externs_dir = msc_utils.msc_dir("msc_externs")
    install_dir = msc_utils.msc_dir("msc_plugins")
    workspace = msc_utils.msc_dir("msc_workspace")
    plugin = _create_plugin(externs_dir)
    managers = build_plugins_manager(
        plugin, [MSCFramework.TORCH], install_dir, externs_dir=externs_dir, workspace=workspace
    )

    model = _get_model()
    # externs_dir.destory()
    # install_dir.destory()
    # workspace.destory()


if __name__ == "__main__":
    # tvm.testing.main()
    test_torch_plugin()
