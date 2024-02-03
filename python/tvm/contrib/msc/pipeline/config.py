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
"""tvm.contrib.msc.pipeline.config"""

from typing import List, Union, Dict

from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils


def config_tool(tool_type, raw_config, **kwargs):
    if isinstance(raw_config, dict):
        tool_style = raw_config.get("tool_style", "default")
    else:
        tool_style, raw_config = raw_config, None
    configer_cls = msc_utils.get_registered_tool_configer(tool_type, tool_style)
    assert configer_cls, "Can not find configer for {}:{}".format(tool_type, tool_style)
    return configer_cls().config(raw_config, **kwargs)


def create_config(
    inputs: List[dict],
    outputs: List[str],
    model_type: str,
    compile_type: str,
    optimize_type: str = None,
    dataset: Dict[str, dict] = None,
    check_accuracy: bool = True,
    prune_config: Union[dict, str] = None,
    quantize_config: Union[dict, str] = None,
    track_config: Union[dict, str] = None,
    distill_config: Union[dict, str] = None,
    gym_configs: Dict[str, Union[dict, str]] = None,
    profile_strategys: Dict[str, Union[dict, str]] = None,
    verbose: str = "info",
    **extra_config,
) -> dict:
    """Create config for msc pipeline

    Parameters
    ----------
    inputs: list<dict>
        The inputs info,
    outputs: list<str>
        The output names.
    model_type: str
        The model type.
    compile_type: str
        The compile type.
    optimize_type: str
        The optimize type.
    dataset: dict<str, dict>
        The datasets for compile pipeline.
    check_accuracy: bool
        Whether to check accuracy.
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
    workspace: str
        The workspace for wrapper.
    verbose: str
        The verbose level for wrapper
    extra_config: dict
        The extra config.
    """

    optimize_type = optimize_type or model_type
    config = {
        "model_type": model_type,
        "verbose": verbose,
        "inputs": inputs,
        "outputs": outputs,
        "dataset": dataset,
        MSCStage.PREPARE: {"profile": {"benchmark": {"repeat": -1}}},
        MSCStage.BASELINE: {
            "run_type": model_type,
            "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
        },
        MSCStage.OPTIMIZE: {
            "run_type": optimize_type,
            "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
        },
        MSCStage.COMPILE: {
            "run_type": compile_type,
            "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
        },
    }
    if not check_accuracy:
        for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]:
            config[stage]["profile"]["check"]["err_rate"] = -1

    # config optimize
    tools_config = {}
    gym_configs = gym_configs or {}
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
        config[MSCStage.OPTIMIZE].update(**tools_config)

    # update profile
    if profile_strategys:
        for stage, config in profile_strategys.items():
            if stage not in config:
                continue
            config[stage]["profile"].update(config)
    if extra_config:
        config = msc_utils.update_dict(config, extra_config)
    return config
