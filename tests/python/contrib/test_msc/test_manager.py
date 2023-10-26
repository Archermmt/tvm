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

""" Test Managers in MSC. """

import pytest
import numpy as np

import torch
from tvm.contrib.msc.framework.tensorflow import tf_v1

import tvm.testing
import tvm.relay.testing.tf as tf_testing
from tvm.contrib.msc.pipeline import MSCManager

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


def _get_torch_model(name, is_training=False):
    """Get model from torch vision and parse to relax"""
    # pylint: disable=import-outside-toplevel
    try:
        import torchvision

        model = getattr(torchvision.models, name)(pretrained=True)
        if is_training:
            model = model.train()
        else:
            model = model.eval()
        return model
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


def _test_from_torch(deploy_type, is_training=False, atol=1e-2, rtol=1e-2):
    torch_model = _get_torch_model("resnet50", is_training)
    if torch_model:
        if torch.cuda.is_available():
            torch_model = torch_model.to(torch.device("cuda:0"))
        config = {
            "inputs": ["input_0", [1, 3, 224, 224], "float32"],
            "outputs": ["output"],
            "dataset": {"loader": "from_random", "config": {"max_iter": 5}},
            "parse": {"type": "torch"},
            "compile": {"type": deploy_type},
            "profile": {
                "baseline": {"benchmark": {"repeat": 10}},
                "parse": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 50}},
                "compile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 50}},
            },
        }
        manager = MSCManager(torch_model, config)
        manager.run_pipe()
        manager.destory()


def test_tvm_manager():
    """Test manager for tvm"""

    _test_from_torch("tvm", is_training=True)


if __name__ == "__main__":
    # tvm.testing.main()
    test_tvm_manager()
