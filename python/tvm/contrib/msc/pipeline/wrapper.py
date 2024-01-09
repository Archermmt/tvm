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


def config_pruner(prune_style: Union[dict, str], gym_style: Union[dict, str], run_type: str):
    """Get the prune config

    Parameters
    ----------
    prune_style: dict/str
        The prune config dict or style.
    gym_style: dict/str
        The gym config dict or style.
    run_type: str
        The runtime type.

    Returns
    -------
    config: dict
        The prune config.
    """

    if isinstance(prune_style, dict):
        config = prune_style
    elif prune_style == "default":
        config = {
            "plan_file": "msc_pruner.json",
            "strategys": [{"method": "per_channel", "density": 0.8}],
        }
    else:
        raise TypeError("Unexpected prune strategy " + str(prune_style))

    if gym_style:
        if isinstance(gym_style, list):
            config["gym_configs"] = gym_style
        elif gym_style == "default":
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
            raise TypeError("Unexpected gym strategy " + str(gym_style))
    return config


def config_quantizer(quantize_style: Union[dict, str], gym_style: Union[dict, str], run_type: str):
    """Get the quantize config

    Parameters
    ----------
    quantize_style: dict/str
        The quantize config dict or style.
    gym_style: dict/str
        The gym config dict or style.
    run_type: str
        The runtime type.

    Returns
    -------
    config: dict
        The quantize config.
    """

    if isinstance(quantize_style, dict):
        config = quantize_style
    elif quantize_style == "default":
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
        raise TypeError("Unexpected quantize strategy " + str(quantize_style))

    if gym_style:
        if isinstance(gym_style, list):
            config["gym_configs"] = gym_style
        elif gym_style == "default":
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
            raise TypeError("Unexpected gym strategy " + str(gym_style))
    return config


def config_tracker(track_style: Union[dict, str], run_type: str):
    """Get the track config

    Parameters
    ----------
    track_style: dict/str
        The track config dict or style.
    run_type: str
        The runtime type.

    Returns
    -------
    config: dict
        The track config.
    """

    if isinstance(track_style, dict):
        return track_style
    if track_style == "default":
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
    raise TypeError("Unexpected track strategy " + str(track_style))


def config_distiller(distill_style: Union[dict, str], run_type: str):
    """Get the distill config

    Parameters
    ----------
    distill_style: dict/str
        The distill config dict or style..
    run_type: str
        The runtime type.

    Returns
    -------
    config: dict
        The distill config.
    """

    if isinstance(distill_style, dict):
        return distill_style
    if distill_style == "default":
        return {
            "plan_file": "msc_distiller.json",
            "strategys": [
                {
                    "method": "loss_lp_norm",
                    "op_types": ["loss"],
                },
            ],
        }
    raise TypeError("Unexpected distill strategy " + str(distill_style))


class BaseWrapper(object):
    """Base Wrapper of models

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    compile_type: str
        The compile type.
    optimize_type: str
        The optimize type.
    dataset: callable
        The data loading method.
    max_batch: int
        The max data batch.
    inputs: list<dict>
        The inputs info,
    outputs: list<str>
        The output names.
    prune_style: dict/str
        The prune config or style.
    quantize_style: dict/str
        The quantize config or style.
    track_style: dict/str
        The track config or style.
    distill_style: dict/str
        The distill config or style.
    gym_styles: dict<str, dict/str>
        The gym configs for tools.
    profile_strategys: dict<str, dict/str>
        The profile configs for tools.
    plugins: dict
        The plugins for pipeline.
    workspace: str
        The workspace.
    debug_leve: int
        The debug level.
    verbose: str
        The verbose level.
    extra_config: dict
        The extra config.
    """

    def __init__(
        self,
        model: Any,
        inputs: List[dict],
        outputs: List[str],
        compile_type: str,
        optimize_type: str = None,
        dataloader: Union[callable, str] = None,
        max_batch: int = -1,
        prune_style: Union[dict, str] = None,
        quantize_style: Union[dict, str] = None,
        track_style: Union[dict, str] = None,
        distill_style: Union[dict, str] = None,
        gym_styles: Dict[str, Union[dict, str]] = None,
        profile_strategys: Dict[str, Union[dict, str]] = None,
        plugins: dict = None,
        workspace: str = "msc_workspace",
        debug_level: int = 0,
        verbose: str = "info",
        **extra_config,
    ):
        self._meta_model = model
        self._optimized_model, self._compiled_model = None, None
        self._dataloader = dataloader
        self._max_batch = max_batch
        self._compile_type = compile_type
        optimize_type = optimize_type or self.model_type
        self._plugins = plugins
        self._manager = None
        self._config = {
            "workspace": workspace,
            "debug_level": debug_level,
            "verbose": verbose,
            "model_type": self.model_type,
            "inputs": inputs,
            "outputs": outputs,
            "dataset": {"loader": dataloader or "from_random", "max_batch": max_batch},
            "prepare": {"profile": {"benchmark": {"repeat": 10}}},
            "baseline": {
                "run_type": self.model_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": 10}},
            },
            "optimize": {
                "run_type": optimize_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": 10}},
            },
            "compile": {
                "run_type": compile_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": 10}},
            },
        }
        # config optimize
        tools_config = {}
        if prune_style:
            tools_config[ToolType.PRUNER] = config_pruner(
                prune_style, gym_styles.get(ToolType.PRUNER), optimize_type
            )
        if quantize_style:
            tools_config[ToolType.QUANTIZER] = config_quantizer(
                quantize_style, gym_styles.get(ToolType.QUANTIZER), optimize_type
            )
        if track_style:
            tools_config[ToolType.TRACKER] = config_tracker(track_style, optimize_type)
        if distill_style:
            tools_config[ToolType.DISTILLER] = config_distiller(distill_style, optimize_type)
        if tools_config:
            self._config["optimize"].update(**tools_config)
        # update profile
        if profile_strategys:
            for stage, config in profile_strategys.items():
                if stage not in self._config:
                    continue
                self._config[stage]["profile"].update(config)
        if extra_config:
            self._config = msc_utils.update_dict(self._config, extra_config)
        self.setup()

    def __str__(self):
        if self.compiled:
            phase = "compiled"
        elif self.optimized:
            phase = "optimized"
        else:
            phase = "meta"
        return "({}) {}".format(phase, self._get_model().__str__())

    def setup(self):
        """Setup the wrapper"""

        pass

    def _create_manager(self):
        if not self._manager:
            self._manager = MSCManager(self._get_model(), self._config, self._plugins)

    def _get_model(self) -> Any:
        return self._compiled_model or self._optimized_model or self._meta_model

    @property
    def optimized(self):
        return self._optimized_model is not None

    @property
    def compiled(self):
        return self._compiled_model is not None

    @property
    def model_type(self):
        return MSCFramework.MSC


class TorchWrapper(object):
    """Wrapper of torch models"""

    @property
    def model_type(self):
        return MSCFramework.TORCH
