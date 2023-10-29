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

import torch
from tvm.contrib.msc.framework.tensorflow import tf_v1

import tvm.testing
import tvm.relay.testing.tf as tf_testing
from tvm.contrib.msc.pipeline import MSCManager
from tvm.contrib.msc.core.utils.namespace import MSCFramework

requires_tensorrt = pytest.mark.skipif(
    tvm.get_global_func("relax.ext.tensorrt", True) is None,
    reason="TENSORRT is not enabled",
)


def _get_config(model_type, deploy_type, inputs, outputs, atol=1e-2, rtol=1e-2):
    """Get msc config"""
    return {
        "model_type": model_type,
        "inputs": inputs,
        "outputs": outputs,
        "dataset": {"loader": "from_random", "max_iter": 5},
        "prepare": {"profile": {"benchmark": {"repeat": 10}}},
        "baseline": {
            "run_type": model_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
        "compile": {
            "run_type": deploy_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
    }


def _get_torch_model(name, is_training=False):
    """Get model from torch vision"""
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


def _get_tf_graph():
    """Get graph from tensorflow"""
    try:
        tf_graph = tf_v1.Graph()
        with tf_graph.as_default():
            graph_def = tf_testing.get_workload(
                "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
                "mobilenet_v2_1.4_224_frozen.pb",
            )
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        return graph_def
    except:
        print("please install tensorflow package")
        return None


def _test_from_torch(deploy_type, is_training=False, atol=1e-2, rtol=1e-2):
    torch_model = _get_torch_model("resnet50", is_training)
    if torch_model:
        if torch.cuda.is_available():
            torch_model = torch_model.to(torch.device("cuda:0"))
        config = _get_config(
            MSCFramework.TORCH,
            deploy_type,
            inputs=[["input_0", [1, 3, 224, 224], "float32"]],
            outputs=["output"],
            atol=atol,
            rtol=rtol,
        )
        manager = MSCManager(torch_model, config)
        report = manager.run_pipe()
        assert report["success"], "Failed to run pipe for torch -> {}".format(deploy_type)
        manager.destory()


def _test_from_tf(deploy_type, atol=1e-2, rtol=1e-2):
    graphdef = _get_tf_graph()
    if graphdef:
        config = _get_config(
            MSCFramework.TENSORFLOW,
            deploy_type,
            inputs=[["input", [1, 224, 224, 3], "float32"]],
            outputs=["MobilenetV2/Predictions/Reshape_1:0"],
            atol=atol,
            rtol=rtol,
        )
        manager = MSCManager(graphdef, config)
        report = manager.run_pipe()
        assert report["success"], "Failed to run pipe for tensorflow -> {}".format(deploy_type)
        manager.destory()


def test_tvm_manager():
    """Test manager for tvm"""

    _test_from_torch(MSCFramework.TVM, is_training=True)
    _test_from_tf(MSCFramework.TVM)


def test_torch_manager():
    """Test manager for torch"""

    _test_from_torch(MSCFramework.TORCH, is_training=False)


def test_tensorflow_manager():
    """Test manager for tensorflow"""

    _test_from_tf(MSCFramework.TENSORFLOW)


@requires_tensorrt
def test_tensorrt_manager():
    """Test manager for tensorrt"""

    _test_from_torch(MSCFramework.TENSORRT, is_training=False)


if __name__ == "__main__":
    tvm.testing.main()
