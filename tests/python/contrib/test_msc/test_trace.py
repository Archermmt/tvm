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

import tvm.testing
from tvm.contrib.msc.core.frontend import msc_trace, get_global_tracer
from tvm.contrib.msc.pipeline import MSCManager, create_config
from tvm.contrib.msc.core import utils as msc_utils


def verify_func(func, input_info, config=None):
    """Compare source function and traced func"""

    config = config or {}
    config.update({"dataset": "trace_datas", "workspace": "trace_workspace", "verbose": "critical"})
    datas = [msc_utils.random_data(i) for i in input_info]
    golden = func(*datas)
    if not isinstance(golden, (tuple, list)):
        golden = [golden]
    traced_func = msc_trace(config)(func)
    _ = traced_func(*datas)
    # dump and compile
    tracer = get_global_tracer()
    info = tracer.dump()
    results = {i: d for i, d in zip(info["inputs"], datas)}
    for group in info["groups"]:
        manager = MSCManager(group["group"]["model"], create_config(**group["group"]["config"]))
        manager.run_pipe()
        runner = manager.get_runtime()
        outputs = runner.run([results[i] for i in group["inputs"]], ret_type="list")
        results.update({o_name: o_data for o_name, o_data in zip(group["outputs"], outputs)})
        manager.destory()
    outputs = [results[o] for o in info["outputs"]]
    for gol, out in zip(golden, outputs):
        tvm.testing.assert_allclose(msc_utils.cast_array(gol), msc_utils.cast_array(out))
    tracer.destory()


def test_binary_ops():
    """Test tracker for binary ops"""

    def common_binary(data):
        data = data + 1
        data = data - 2
        data = data * data
        data /= 5
        data *= 2
        data = data**2
        return data

    verify_func(common_binary, [([256, 256], "float32")])


if __name__ == "__main__":
    # tvm.testing.main()
    test_binary_ops()
