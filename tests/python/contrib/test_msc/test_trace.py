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

""" Test trace in MSC. """

import numpy as np

import tvm.testing
from tvm.contrib.msc.core.frontend import msc_trace, get_global_tracer
from tvm.contrib.msc.pipeline import MSCManager, create_config
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.torch.frontend.trace import *


def _create_manager(group: dict) -> MSCManager:
    print("group " + str(group))
    config = create_config(
        group["inputs"],
        group["outputs"],
        group["model_type"],
        verbose="critical",
        workspace="{}_workspace".format(group["name"]),
        dataset={MSCStage.PREPARE: {"loader": group["dataset"]}},
    )
    return MSCManager(group["model"], config)


def verify_func(func, input_info, framework=MSCFramework.MSC, config=None):
    """Compare source function and traced func"""

    config = config or {}
    config.update({"dataset": "trace_datas", "workspace": "trace_workspace", "verbose": "critical"})

    def _to_data(i_info):
        if isinstance(i_info, np.ndarray):
            return msc_utils.cast_array(i_info, framework)
        return msc_utils.random_data(i_info, framework)

    print("\n\n[TMINFO] start tracing func " + str(func))
    datas = [_to_data(i) for i in input_info]
    golden = func(*datas)
    if not isinstance(golden, (tuple, list)):
        golden = [golden]
    for idx, g in enumerate(golden):
        print("{} th golden {}".format(idx, msc_utils.inspect_array(g)))
    traced_func = msc_trace(config)(func)
    _ = traced_func(*datas)
    # dump and compile
    tracer = get_global_tracer()
    dumped, managers = tracer.dump(), {}
    for group in dumped["groups"]:
        managers[group["name"]] = _create_manager(group)
        managers[group["name"]].run_pipe()
    # run graph
    results = {i: d for i, d in zip(dumped["inputs"], datas)}
    for group in dumped["groups"]:
        runner = managers[group["name"]].get_runtime()
        outputs = runner.run([results[i[0]] for i in group["inputs"]], ret_type="list")
        results.update({o_name: o_data for o_name, o_data in zip(group["outputs"], outputs)})
    outputs = [results[o] for o in dumped["outputs"]]
    for gol, out in zip(golden, outputs):
        tvm.testing.assert_allclose(msc_utils.cast_array(gol), msc_utils.cast_array(out))
    # clean up
    dumped["dataset"].destory()
    for manager in managers.values():
        manager.destory()
    print("[TMINFO] destory!!")


def test_binary():
    """Test trace for binary"""

    def func_binary(data):
        data = data + 1
        data = data - 2
        data = data * data
        data /= 5
        data *= 2
        data = data**2
        return data

    def func_numpy(data_a, data_b):
        data_max = np.maximum(data_a, data_b)
        data_min = np.minimum(data_a, data_b)
        return data_max * data_min

    verify_func(func_binary, [([256, 256], "float32")])
    verify_func(func_numpy, [([256, 256], "float32"), ([256, 256], "float32")])


def test_unary():
    """Test trace for unary"""

    def func_numpy(data):
        data = data.abs()
        return np.exp(data)

    verify_func(func_numpy, [([256, 256], "float32")])


def test_compare():
    """Test trace for compare"""

    def func_compare(data_a, data_b):
        data_eq = data_a == data_b
        data_ne = data_a != data_b
        data_le = data_a <= data_b
        data_lt = data_a < data_b
        data_ge = data_a >= data_b
        data_gt = data_a > data_b
        return data_eq * data_ne * data_le * data_lt * data_ge * data_gt

    verify_func(func_compare, [([256, 256], "float32"), ([256, 256], "float32")])


def test_getitem():
    """Test trace for getitem"""

    def func_slice(data):
        data = data[0]
        data = data[1::2, :, :3]
        return data[0, :, None]

    def func_select(data, index1, index2):
        data = data[index1]
        return data[:, index2]

    verify_func(func_slice, [([1, 3, 10, 10], "float32")])
    verify_func(func_select, [([256, 256], "float32"), ([10], "int32"), ([10], "int32")])


def test_setitem():
    """Test trace for getitem"""

    def func_put(data, value, index):
        data[index] = value
        return data

    def func_mask_scatter(data, value, mask):
        data[mask] = value
        return data

    mask = np.array([True, True, False, False, True, False, True, False, True, False])
    for framework in [MSCFramework.MSC, MSCFramework.TORCH]:
        verify_func(
            func_put, [([256, 256], "float32"), ([10, 256], "float32"), ([10], "int32")], framework
        )
        verify_func(
            func_mask_scatter, [([10, 256], "float32"), ([5, 256], "float32"), mask], framework
        )


if __name__ == "__main__":
    # tvm.testing.main()
    test_binary()
