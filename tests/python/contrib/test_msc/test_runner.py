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

""" Test Runners in MSC. """

import pytest
import numpy as np

import torch
from torch import fx
from tvm.contrib.msc.framework.tensorflow import tf_v1

import tvm.testing
import tvm.relay.testing.tf as tf_testing
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.framework.tvm.runtime import TVMRunner
from tvm.contrib.msc.framework.torch.runtime import TorchRunner
from tvm.contrib.msc.framework.tensorrt.runtime import TensorRTRunner
from tvm.contrib.msc.framework.tensorflow.frontend import from_tensorflow
from tvm.contrib.msc.framework.tensorflow.runtime import TensorflowRunner
from tvm.contrib.msc.core import utils as msc_utils

requires_tensorrt = pytest.mark.skipif(
    tvm.get_global_func("relax.ext.tensorrt", True) is None,
    reason="TENSORRT is not enabled",
)


def _get_module_from_torch(name):
    """Get model from torch vision and parse to relax"""
    # pylint: disable=import-outside-toplevel
    try:
        import torchvision

        return getattr(torchvision.models, name)()
    except:  # pylint: disable=bare-except
        print("please install torchvision package")
        return None


def _get_graph_from_tf():
    try:
        tf_graph = tf_v1.Graph()
        with tf_graph.as_default():
            graph_def = tf_testing.get_workload(
                "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
                "mobilenet_v2_1.4_224_frozen.pb",
            )
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        return tf_graph, graph_def
    except:
        print("please install tensorflow package")
        return None, None


def _test_runner_from_torch(runner_cls, device):
    torch_model = _get_module_from_torch("resnet50")
    if torch_model:
        workspace = msc_utils.set_workspace()
        graph_model = fx.symbolic_trace(torch_model)
        input_info = [([1, 3, 224, 224], "float32")]
        datas = [np.random.rand(*i[0]).astype(i[1]) for i in input_info]
        torch_datas = [torch.from_numpy(d) for d in datas]
        with torch.no_grad():
            golden = torch_model(*torch_datas)
            mod = from_fx(graph_model, input_info)
        runner = runner_cls(mod, device=device)
        runner.build()
        outputs = runner.run(datas, ret_type="list")
        golden = [msc_utils.cast_array(golden)]
        for gol_r, out_r in zip(golden, outputs):
            tvm.testing.assert_allclose(gol_r, out_r, atol=1e-3, rtol=1e-3)
        workspace.destory()


def test_tvm_runner_cpu():
    """Test runner for tvm on cpu"""

    _test_runner_from_torch(TVMRunner, "cpu")


@tvm.testing.requires_gpu
def test_tvm_runner_gpu():
    """Test runner for tvm on gpu"""

    _test_runner_from_torch(TVMRunner, "gpu")


def test_torch_runner_cpu():
    """Test runner for torch on cpu"""

    _test_runner_from_torch(TorchRunner, "cpu")


@tvm.testing.requires_gpu
def test_torch_runner_gpu():
    """Test runner for torch on gpu"""

    _test_runner_from_torch(TorchRunner, "gpu")


# @requires_tensorrt
def test_tensorrt_runner():
    """Test runner for tensorrt"""

    _test_runner_from_torch(TensorRTRunner, "gpu")


def test_tensorflow_runner():
    tf_graph, graph_def = _get_graph_from_tf()
    if tf_graph and graph_def:
        workspace = msc_utils.set_workspace()
        data = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")
        out_name = "MobilenetV2/Predictions/Reshape_1:0"
        # get golden
        with tf_v1.Session(graph=tf_graph) as sess:
            golden = sess.run([out_name], {"input:0": data})
        # get outputs
        shape_dict = {"input": data.shape}
        mod, _ = from_tensorflow(graph_def, shape_dict, [out_name], as_msc=False)
        runner = TensorflowRunner(mod)
        runner.build()
        outputs = runner.run([data], ret_type="list")
        for gol_r, out_r in zip(golden, outputs):
            tvm.testing.assert_allclose(gol_r, out_r, atol=1e-3, rtol=1e-3)
        workspace.destory()


if __name__ == "__main__":
    # tvm.testing.main()
    test_tensorrt_runner()
