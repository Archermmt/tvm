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

import shutil
from typing import List, Any, Union, Dict

from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.message import MSCStage
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
    dataset: dict<str, dict>
        The datasets for compile pipeline.
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
    workspace: str
        The workspace for wrapper.
    verbose: str
        The verbose level for wrapper
    debug: bool
        Whether to use debug mode.
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
        dataset: Dict[str, dict] = None,
        check_baseline: bool = True,
        prune_config: Union[dict, str] = None,
        quantize_config: Union[dict, str] = None,
        track_config: Union[dict, str] = None,
        distill_config: Union[dict, str] = None,
        gym_configs: Dict[str, Union[dict, str]] = None,
        profile_strategys: Dict[str, Union[dict, str]] = None,
        plugins: dict = None,
        workspace: str = "msc_workspace",
        verbose: str = "info",
        debug: bool = False,
        **extra_config,
    ):
        self._meta_model = model
        self._optimized_model, self._compiled_model = None, None
        self._optimize_type = optimize_type or self.model_type
        self._compile_type = compile_type
        self._plugins = plugins
        self._workspace = msc_utils.msc_dir(workspace, keep_history=debug)
        log_path = self._workspace.relpath("MSC_LOG", keep_history=False)
        self._logger = msc_utils.create_file_logger(verbose, log_path)
        self._debug = debug
        self._manager = None
        self._config = {
            "model_type": self.model_type,
            "versboe": verbose,
            "logger": self._logger,
            "inputs": inputs,
            "outputs": outputs,
            "dataset": dataset,
            MSCStage.PREPARE: {"profile": {"benchmark": {"repeat": -1}}},
            MSCStage.OPTIMIZE: {
                "run_type": self._optimize_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
            },
            MSCStage.COMPILE: {
                "run_type": compile_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
            },
        }
        if check_baseline:
            self._config[MSCStage.BASELINE] = {
                "run_type": self.model_type,
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
            }

        # config optimize
        self._tools_config = {}
        gym_configs = gym_configs or {}
        if prune_config:
            self._tools_config[ToolType.PRUNER] = config_tool(
                ToolType.PRUNER, prune_config, gym_configs=gym_configs.get(ToolType.PRUNER)
            )
        if quantize_config:
            self._tools_config[ToolType.QUANTIZER] = config_tool(
                ToolType.QUANTIZER, quantize_config, gym_configs=gym_configs.get(ToolType.QUANTIZER)
            )
        if track_config:
            self._tools_config[ToolType.TRACKER] = config_tool(ToolType.TRACKER, track_config)
        if distill_config:
            self._tools_config[ToolType.DISTILLER] = config_tool(ToolType.DISTILLER, distill_config)
        if self._tools_config:
            self._config[MSCStage.OPTIMIZE].update(**self._tools_config)

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

    def optimize(self, workspace: str = "Optimize"):
        """Optimize the model

        Parameters
        ----------
        workspace: str
            The workspace.
        """

        self._logger.info("[Wrapper] Start optimize model")
        config = msc_utils.copy_dict(self._config)
        config["workspace"] = self._workspace.create_dir(workspace)
        self._manager = MSCManager(self._meta_model, config, self._plugins)
        self._manager.run_pipe(run_compile=False)
        self._optimized_model = self._manager.get_runnable("runnable")

    def compile(self, workspace: str = "Compile", ckpt_path: str = "Checkpoint"):
        """Compile the model

        Parameters
        ----------
        workspace: str
            The workspace.
        ckpt_path: str
            The path to export checkpoint.
        """

        if self._optimized_model:
            self._logger.info("[Wrapper] Start compile checkpoint")
            ckpt_path = self._workspace.create_dir(ckpt_path).path
            pipeline = self.export(ckpt_path, dump=False)
            pipeline["config"]["workspace"] = self._workspace.create_dir(workspace)
            self._manager = MSCManager(**pipeline)
            self._manager.run_pipe(run_optimize=False)
            self._compiled_model = self._manager.get_runnable("runnable")
            if not self._debug:
                shutil.rmtree(ckpt_path)
        else:
            self._logger.info("[Wrapper] Start compile model")
            config = msc_utils.copy_dict(self._config)
            config["workspace"] = self._workspace.create_dir(workspace)
            self._manager = MSCManager(self._meta_model, config, self._plugins)
            self._manager.run_pipe()
            self._compiled_model = self._manager.get_runnable("runnable")

    def export(self, path: str, dump: bool = True) -> Union[str, dict]:
        """Export compile pipeline

        Parameters
        ----------
        path: str
            The export path.
        dump: bool
            Whether to dump the info.

        Returns
        -------
        export_path/pipeline: str/dict
            The exported path/pipeline info.
        """

        assert self._manager, "manager is needed to export wrapper"
        exported = self._manager.export(path, dump=dump)
        if not self._debug:
            self._manager.destory()
        return exported

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

    def __call__(self, *inputs):
        framework = self._get_framework()
        if framework != MSCFramework.TORCH:
            inputs = [msc_utils.cast_array(i, framework) for i in inputs]
        outputs = self._get_model()(*inputs)
        if framework == MSCFramework.TORCH:
            return outputs
        if isinstance(outputs, (tuple, list)):
            return [msc_utils.cast_array(o, MSCFramework.TORCH) for o in outputs]
        return msc_utils.cast_array(outputs, MSCFramework.TORCH)

    def parameters(self):
        framework = self._get_framework()
        if framework == MSCFramework.TORCH:
            res = self._get_model().parameters()
            print("normal parmeters " + str(res))
            return res
        for w in self._manager.runner.get_weights():
            print("has weight {}({})".format(w, type(w)))
        raise Exception("stop here!!")

    def _get_framework(self) -> str:
        return self._manager.runner.framework if self._manager else MSCFramework.TORCH

    @property
    def model_type(self):
        return MSCFramework.TORCH
