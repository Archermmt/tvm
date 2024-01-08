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
"""tvm.contrib.msc.pipeline.wrapper"""

from typing import List, Any, Union, Dict

from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from .manager import MSCManager


def config_pruner(prune_strategy: Union[dict, str], gym_strategy: Union[dict, str], run_type: str):
    """Get the prune config

    Parameters
    ----------
    prune_strategy: dict/str
        The strategy type or config.
    gym_strategy: dict/str
        The gym type or config.
    run_type: str
        The runtime type for quantizer.

    Returns
    -------
    config: dict
        The prune config.
    """

    if isinstance(prune_strategy, dict):
        config = prune_strategy
    elif prune_strategy == "default":
        config = {
            "plan_file": "msc_pruner.json",
            "strategys": [{"method": "per_channel", "density": 0.8}],
        }
    else:
        raise TypeError("Unexpected prune strategy " + str(prune_strategy))

    if gym_strategy:
        if isinstance(gym_strategy, list):
            config["gym_configs"] = gym_strategy
        elif gym_strategy == "default":
            config["gym_configs"] = (
                [
                    {
                        "env": {
                            "executors": {
                                "action_space": {
                                    "method": "action_prune_density",
                                    "start": 0.2,
                                    "end": 0.8,
                                    "step": 0.1,
                                }
                            },
                        },
                        "agent": {"agent_type": "search.grid", "executors": {}},
                    }
                ],
            )
        else:
            raise TypeError("Unexpected gym strategy " + str(gym_strategy))
    return config


def config_quantizer(
    quantize_strategy: Union[dict, str], gym_strategy: Union[dict, str], run_type: str
):
    """Get the quantize config

    Parameters
    ----------
    quantize_strategy: dict/str
        The strategy type or config.
    gym_strategy: dict/str
        The gym type or config.
    run_type: str
        The runtime type for quantizer.

    Returns
    -------
    config: dict
        The quantize config.
    """

    if isinstance(quantize_strategy, dict):
        config = quantize_strategy
    elif quantize_strategy == "default":
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.msc.core.tools.quantize import QuantizeStage

        if run_type == MSCFramework.TENSORRT:
            config = {"plan_file": "msc_quantizer.json"}
        else:
            config = {
                "plan_file": "msc_quantizer.json",
                "strategys": [
                    {
                        "method": "gather_maxmin",
                        "op_types": ["nn.conv2d", "msc.linear"],
                        "tensor_types": ["input", "output"],
                        "stages": [QuantizeStage.GATHER],
                    },
                    {
                        "method": "gather_max_per_channel",
                        "op_types": ["nn.conv2d", "msc.linear"],
                        "tensor_types": ["weight"],
                        "stages": [QuantizeStage.GATHER],
                    },
                    {
                        "method": "calibrate_maxmin",
                        "op_types": ["nn.conv2d", "msc.linear"],
                        "tensor_types": ["input", "output"],
                        "stages": [QuantizeStage.CALIBRATE],
                    },
                    {
                        "method": "quantize_normal",
                        "op_types": ["nn.conv2d", "msc.linear"],
                        "tensor_types": ["input", "weight"],
                    },
                    {
                        "method": "dequantize_normal",
                        "op_types": ["nn.conv2d", "msc.linear"],
                        "tensor_types": ["output"],
                    },
                ],
            }
    else:
        raise TypeError("Unexpected quantize strategy " + str(quantize_strategy))

    if gym_strategy:
        if isinstance(gym_strategy, list):
            config["gym_configs"] = gym_strategy
        elif gym_strategy == "default":
            config["gym_configs"] = (
                [
                    {
                        "env": {
                            "executors": {
                                "action_space": {
                                    "method": "action_quantize_scale",
                                    "start": 0.8,
                                    "end": 1.2,
                                    "step": 0.1,
                                }
                            },
                        },
                        "agent": {"agent_type": "search.grid", "executors": {}},
                    }
                ],
            )
        else:
            raise TypeError("Unexpected gym strategy " + str(gym_strategy))
    return config


def config_tracker(track_strategy: Union[dict, str], run_type: str):
    """Get the track config

    Parameters
    ----------
    track_strategy: dict/str
        The strategy type or config.
    run_type: str
        The runtime type for quantizer.

    Returns
    -------
    config: dict
        The prune config.
    """

    if isinstance(track_strategy, dict):
        return track_strategy
    if track_strategy == "default":
        return {
            "plan_file": "msc_tracker.json",
            "strategys": [
                {
                    "method": "save_compared",
                    "compare_to": {
                        "optimize": ["baseline"],
                        "compile": ["optimize", "baseline"],
                    },
                    "op_types": ["nn.relu"],
                    "tensor_types": ["output"],
                }
            ],
        }
    raise TypeError("Unexpected track strategy " + str(track_strategy))


class BaseWrapper(object):
    """Base Wrapper of models

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    plugins: dict
        The plugins fro pipeline.
    """

    def __init__(
        self,
        model: Any,
        compile_type: str,
        optimize_type: str = None,
        dataset: Union[callable, str] = None,
        max_batch: int = -1,
        inputs: List[dict] = None,
        outputs: List[str] = None,
        prune_strategy: Union[dict, str] = None,
        quantize_strategy: Union[dict, str] = None,
        track_strategy: Union[dict, str] = None,
        distill_strategy: Union[dict, str] = None,
        gym_strategys: Dict[str, Union[dict, str]] = None,
        workspace: str = "msc_workspace",
        debug_level: int = 0,
        verbose: str = "info",
        profile_strategy: Dict[str, Union[dict, str]] = None,
        **extra_config,
    ):
        print("wrap the model " + str(model))
        self._config = {
            "workspace": workspace,
            "debug_level": debug_level,
            "verbose": verbose,
            "model_type": self.model_type,
            "inputs": inputs,
            "outputs": outputs,
            "dataset": {"loader": dataset or "from_random", "max_batch": max_batch},
            "prepare": {"profile": {"benchmark": {"repeat": 10}}},
            "baseline": {
                "run_type": self.model_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": 10}},
            },
            "compile": {
                "run_type": compile_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": 10}},
            },
        }
        tools_config = {}
        optimize_type = optimize_type or self.model_type
        if prune_strategy:
            tools_config[ToolType.PRUNER] = config_pruner(
                prune_strategy, gym_strategys.get(ToolType.PRUNER), optimize_type
            )
        if quantize_strategy:
            tools_config[ToolType.QUANTIZER] = config_quantizer(
                quantize_strategy, gym_strategys.get(ToolType.QUANTIZER), optimize_type
            )
        if track_strategy:
            tools_config[ToolType.TRACKER] = config_tracker(track_strategy, optimize_type)

        self._compile_type = compile_type
        self._dataset = dataset
        self._inputs = inputs
        self._outputs = outputs
        self._optimize_type = optimize_type or self.model_type

    @property
    def model_type(self):
        return MSCFramework.MSC
