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


def config_tool(tool_type, raw_config, **kwargs):
    if isinstance(raw_config, dict):
        tool_style = raw_config.get("tool_style", "default")
    else:
        tool_style, raw_config = raw_config, None
    configer_cls = msc_utils.get_registered_tool_configer(tool_type, tool_style)
    assert configer_cls, "Can not find configer for {}:{}".format(tool_type, tool_style)
    return configer_cls().config(raw_config, **kwargs)


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
    check_baseline: bool
        Whether to check baseline.
    inputs: list<dict>
        The inputs info,
    outputs: list<str>
        The output names.
    prune_config: dict/str
        The prune config or style.
    quantize_config: dict/str
        The quantize config or style.
    track_config: dict/str
        The track config or style.
    distill_config: dict/str
        The distill config or style.
    gym_configs: dict<str, dict/str>
        The gym configs for tools.
    profile_strategys: dict<str, dict/str>
        The profile configs for tools.
    plugins: dict
        The plugins for pipeline.
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
        check_baseline: bool = True,
        prune_config: Union[dict, str] = None,
        quantize_config: Union[dict, str] = None,
        track_config: Union[dict, str] = None,
        distill_config: Union[dict, str] = None,
        gym_configs: Dict[str, Union[dict, str]] = None,
        profile_strategys: Dict[str, Union[dict, str]] = None,
        plugins: dict = None,
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
            "model_type": self.model_type,
            "inputs": inputs,
            "outputs": outputs,
            "dataset": {"loader": dataloader or "from_random", "max_batch": max_batch},
            "prepare": {"profile": {"benchmark": {"repeat": -1}}},
            "optimize": {
                "run_type": optimize_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
            },
            "compile": {
                "run_type": compile_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
            },
        }
        if check_baseline:
            self._config["baseline"] = {
                "run_type": self.model_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
            }

        # config optimize
        tools_config = {}
        if prune_config:
            tools_config[ToolType.PRUNER] = config_tool(
                ToolType.PRUNER, prune_config, gym_configs=gym_configs.get(ToolType.PRUNER)
            )
        if quantize_config:
            tools_config[ToolType.QUANTIZER] = config_tool(
                ToolType.QUANTIZER, quantize_config, gym_configs=gym_configs.get(ToolType.QUANTIZER)
            )
        if track_config:
            tools_config[ToolType.TRACKER] = config_tool(ToolType.TRACKER, track_config)
        if distill_config:
            tools_config[ToolType.DISTILLER] = config_tool(ToolType.DISTILLER, distill_config)
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

    def __getattr__(self, name):
        if hasattr(self._get_model(), name):
            return getattr(self._get_model(), name)
        return self._get_model().__getattr__(name)

    def setup(self):
        """Setup the wrapper"""

        pass

    def optimize(self):
        """Optimize the model"""

        self._create_manager()
        self._manager.prepare()
        self._manager.parse()
        if "baseline" in self._config:
            self._manager.baseline()
        self._manager.optimize()
        self._optimized_model = self._manager.get_runnable("runnable")

    def compile(self):
        self._create_manager()
        if not self._optimized_model:
            self.optimize()
        self._compiled_model = self._manager.get_runnable("runnable")

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


class TorchWrapper(BaseWrapper):
    """Wrapper of torch models"""

    @property
    def model_type(self):
        return MSCFramework.TORCH
