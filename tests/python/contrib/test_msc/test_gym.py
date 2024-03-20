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

""" Test Gym in MSC. """

import json
import pytest
import torch

import tvm.testing
from tvm.contrib.msc.pipeline import MSCManager
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils

requires_tensorrt = pytest.mark.skipif(
    tvm.get_global_func("relax.ext.tensorrt", True) is None,
    reason="TENSORRT is not enabled",
)


def _get_config(
    model_type,
    compile_type,
    tools,
    inputs,
    outputs,
    atol=1e-2,
    rtol=1e-2,
    optimize_type=None,
):
    """Get msc config"""

    path = "_".join(["test_gym", model_type, compile_type] + [t["tool_type"] for t in tools])
    return {
        "workspace": msc_utils.msc_dir(path),
        "verbose": "debug:3",
        "model_type": model_type,
        "inputs": inputs,
        "outputs": outputs,
        "dataset": {"prepare": {"loader": "from_random", "max_iter": 5}},
        "tools": tools,
        "prepare": {"profile": {"benchmark": {"repeat": 10}}},
        "baseline": {
            "run_type": model_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
        "optimize": {
            "run_type": optimize_type or model_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
        "compile": {
            "run_type": compile_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
    }


def get_tools(tool_type):
    """Get config for the tool"""

    tools = []
    if tool_type == ToolType.PRUNER:
        config = {
            "plan_file": "msc_pruner.json",
            "strategys": [
                {
                    "methods": {
                        "weights": {"method_name": "per_channel", "density": 0.8},
                        "output": {"method_name": "per_channel", "density": 0.8},
                    }
                }
            ],
        }
        gym_configs = [
            {
                "env": {
                    "executors": {
                        "action_space": {
                            "method": "action_prune_density",
                            "start": 0.4,
                            "end": 0.8,
                            "step": 0.4,
                        }
                    },
                    "max_tasks": 3,
                },
                "agent": {"role_type": "search.grid", "executors": {}},
            }
        ]
        tools.append(
            {"tool_type": ToolType.PRUNER, "tool_config": config, "gym_configs": gym_configs}
        )
    elif tool_type == ToolType.QUANTIZER:
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.msc.core.tools.quantize import QuantizeStage

        config = {
            "plan_file": "msc_quantizer.json",
            "strategys": [
                {
                    "methods": {
                        "input": "gather_maxmin",
                        "output": "gather_maxmin",
                        "weight": "gather_max_per_channel",
                    },
                    "op_types": ["nn.conv2d", "msc.linear"],
                    "stages": [QuantizeStage.GATHER],
                },
                {
                    "methods": {"input": "calibrate_maxmin", "output": "calibrate_maxmin"},
                    "op_types": ["nn.conv2d", "msc.linear"],
                    "stages": [QuantizeStage.CALIBRATE],
                },
                {
                    "methods": {
                        "input": "quantize_normal",
                        "weight": "quantize_normal",
                        "output": "dequantize_normal",
                    },
                    "op_types": ["nn.conv2d", "msc.linear"],
                },
            ],
        }
        gym_configs = [
            {
                "env": {
                    "executors": {
                        "action_space": {
                            "method": "action_quantize_scale",
                            "start": 0.8,
                            "end": 1.2,
                            "step": 0.2,
                        }
                    },
                    "max_tasks": 3,
                },
                "agent": {"role_type": "search.grid", "executors": {}},
            }
        ]
        tools.append(
            {"tool_type": ToolType.QUANTIZER, "tool_config": config, "gym_configs": gym_configs}
        )
    return tools


def _get_torch_model(name, training=False):
    """Get model from torch vision"""

    # pylint: disable=import-outside-toplevel
    try:
        import torchvision

        model = getattr(torchvision.models, name)()
        if training:
            model = model.train()
        else:
            model = model.eval()
        return model
    except:  # pylint: disable=bare-except
        print("please install torchvision package")
        return None


def _check_manager(manager, expected_info):
    """Check the manager results"""

    model_info = manager.get_runtime().model_info
    passed, err = True, ""
    if not manager.report["success"]:
        passed = False
        err = "Failed to run pipe for {} -> {}".format(manager.model_type, manager.compile_type)
    if not msc_utils.dict_equal(model_info, expected_info):
        passed = False
        err = "Model info {} mismatch with expected {}".format(model_info, expected_info)
    # manager.destory()
    if not passed:
        raise Exception("{}\nReport:{}".format(err, json.dumps(manager.report, indent=2)))


def _test_from_torch(
    compile_type,
    tools,
    expected_info,
    training=False,
    atol=1e-1,
    rtol=1e-1,
    optimize_type=None,
):
    torch_model = _get_torch_model("resnet50", training)
    if torch_model:
        if torch.cuda.is_available():
            torch_model = torch_model.to(torch.device("cuda:0"))
        config = _get_config(
            MSCFramework.TORCH,
            compile_type,
            tools,
            inputs=[["input_0", [1, 3, 224, 224], "float32"]],
            outputs=["output"],
            atol=atol,
            rtol=rtol,
            optimize_type=optimize_type,
        )
        manager = MSCManager(torch_model, config)
        manager.run_pipe()
        _check_manager(manager, expected_info)


def get_model_info(compile_type):
    """Get the model info"""

    if compile_type == MSCFramework.TVM:
        return {
            "inputs": [
                {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
            ],
            "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": "NC"}],
            "nodes": {
                "total": 229,
                "input": 1,
                "nn.conv2d": 53,
                "nn.batch_norm": 53,
                "get_item": 53,
                "nn.relu": 49,
                "nn.max_pool2d": 1,
                "add": 16,
                "nn.adaptive_avg_pool2d": 1,
                "reshape": 1,
                "msc.linear_bias": 1,
            },
        }
    if compile_type == MSCFramework.TENSORRT:
        return {
            "inputs": [
                {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
            ],
            "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": ""}],
            "nodes": {"total": 2, "input": 1, "msc_tensorrt": 1},
        }
    raise TypeError("Unexpected compile_type " + str(compile_type))


@pytest.mark.parametrize("tool_type", [ToolType.PRUNER, ToolType.QUANTIZER])
def test_tvm_gym(tool_type):
    """Test tools for tvm with gym"""

    tools = get_tools(tool_type)
    _test_from_torch(MSCFramework.TVM, tools, get_model_info(MSCFramework.TVM), training=True)


@requires_tensorrt
@pytest.mark.parametrize("tool_type", [ToolType.PRUNER])
def test_tensorrt_gym(tool_type):
    """Test tools for tensorrt with gym"""

    tools = get_tools(tool_type)
    _test_from_torch(
        MSCFramework.TENSORRT, tools, get_model_info(MSCFramework.TENSORRT), training=False
    )


if __name__ == "__main__":
    # tvm.testing.main()
    test_tvm_gym(ToolType.PRUNER)
