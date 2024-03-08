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
"""tvm.contrib.msc.pipeline.dynamic"""

import os
import time
import json
from typing import Dict, Any, Union, List
import traceback
import numpy as np

import tvm
from tvm.contrib.msc.core.runtime import BaseRunner
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework, MSCMap, MSCKey
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core.gym.control import create_controller
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.plugin.utils import export_plugins, load_plugins
from .config import support_tool


class BaseDynamic(object):
    """Base Dynamic of MSC

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    plugins: dict
        The plugins for pipeline.
    root: str
        The root path for files.
    run_optimize: bool
        Whether to run optimize.
    run_compile: bool
        Whether to run compile.
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        plugins: dict = None,
        root: str = None,
        run_optimize: bool = True,
        run_compile: bool = True,
    ):
        # change path to root path
        if root:

            def _from_root_mark(val):
                if isinstance(val, str) and MSCKey.ROOT_MARK in val:
                    return val.replace(MSCKey.ROOT_MARK, root)
                return val

            model = _from_root_mark(model)
            config = msc_utils.map_dict(config, _from_root_mark)
            plugins = msc_utils.map_dict(plugins, _from_root_mark)

        # check stage
        for stage in ["dataset", MSCStage.PREPARE, MSCStage.PARSE, MSCStage.COMPILE]:
            config.setdefault(stage, {})

        MSCMap.reset()
        use_cache = config.get("use_cache", True)
        self._workspace = msc_utils.set_workspace(config.get("workspace"), use_cache)
        self._model_type = config["model_type"]
        self._model = model
        self._plugins = load_plugins(plugins) if plugins else {}
        self._verbose = config.get("verbose", "info")
        if "logger" in config:
            self._logger = config["logger"]
            MSCMap.set(MSCKey.GLOBALE_LOGGER, self._logger)
        else:
            log_path = config.get("log_path") or self._workspace.relpath(
                "MSC_LOG", keep_history=False
            )
            self._logger = msc_utils.set_global_logger(self._verbose, log_path)
        self._optimized, self._compiled = False, False
        msc_utils.time_stamp(MSCStage.SETUP)
        self._logger.info(
            msc_utils.msg_block("SETUP", self.setup(config, run_optimize, run_compile))
        )

    def setup(self, config: dict, run_optimize: bool = True, run_compile: bool = True) -> dict:
        """Setup the dynamic

        Parameters
        ----------
        config: dict
            The config for manager.
        run_optimize: bool
            Whether to run optimize.
        run_compile: bool
            Whether to run compile.

        Returns
        -------
        info: dict
            The setup info.
        """

        self._meta_config = config
        self._optimize_type = config.get(MSCStage.OPTIMIZE, {}).get("run_type", self._model_type)
        self._compile_type = config.get(MSCStage.COMPILE, {}).get("run_type", self._model_type)
        # register plugins
        if self._plugins:
            for t in [self._model_type, self._optimize_type, self._compile_type]:
                assert t in self._plugins, "Missing plugin for {}".format(t)
            for name, plugin in self._plugins[self._model_type].get_ops_info().items():
                _ffi_api.RegisterPlugin(name, msc_utils.dump_dict(plugin))
        self._common_config, self._debug_levels = self.update_config(config)
        if not run_optimize and MSCStage.OPTIMIZE in self._common_config:
            self._common_config.pop(MSCStage.OPTIMIZE)
        if not run_compile and MSCStage.COMPILE in self._common_config:
            self._common_config.pop(MSCStage.COMPILE)
        self._report = {
            "success": False,
            "info": {
                "workspace": self._workspace.path,
                "model_type": self._model_type,
            },
            "duration": {},
            "profile": {},
        }
        return {
            "workspace": self._workspace.path,
            "plugins": self._plugins,
            "common_config": self._common_config,
        }

    def update_config(self, config: dict) -> dict:
        """Update config

        Parameters
        ----------
        config: dict
            The config for manager.

        Returns
        -------
        config: dict
            The updated config.
        """

        config, debug_levels = msc_utils.copy_dict(config), {}

        def _set_debug_level(stage: str, sub_config: dict, default: int = None) -> dict:
            if "debug_level" in sub_config:
                debug_levels[stage] = sub_config["debug_level"]
            elif default is not None:
                debug_levels[stage] = default
                sub_config["debug_level"] = default
            return debug_levels

        if self._verbose.startswith("debug:"):
            debug_level = int(self._verbose.split(":")[1])
        else:
            debug_level = 0
        for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]:
            if stage not in config:
                continue
            debug_levels = _set_debug_level(stage, config[stage]["run_config"], debug_level)
            for t_config in config.get("tools", []):
                if not support_tool(t_config, stage, config[stage]["run_type"]):
                    continue
                t_stage = stage + "." + self._get_tool_stage(t_config["tool_type"])
                debug_levels = _set_debug_level(t_stage, t_config["tool_config"], debug_level)
        ordered_keys = [
            "model_type",
            "dataset",
            "tools",
            MSCStage.PREPARE,
            MSCStage.PARSE,
            MSCStage.BASELINE,
            MSCStage.OPTIMIZE,
            MSCStage.COMPILE,
        ]
        return {k: config[k] for k in ordered_keys if k in config}, debug_levels

    def run_pipe(self) -> dict:
        """Run the pipeline and return object.

        Returns
        -------
        report:
            The pipeline report.
        """

        err_msg = None
        try:
            self.prepare()
            self.parse()
            if MSCStage.BASELINE in self._common_config:
                self.baseline()
            if MSCStage.OPTIMIZE in self._common_config:
                self.optimize()
            if MSCStage.COMPILE in self._common_config:
                self.compile()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            err_msg = "Pipeline failed:{}\nTrace: {}".format(exc, traceback.format_exc())
        self.summary(err_msg)
        self._logger.info(msc_utils.msg_block("SUMMARY", self._report, 0))
        self._workspace.finalize()
        return self._report

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

    def __call__(self, *inputs):
        return self._get_model()(*inputs)

    def setup(self):
        """Setup the wrapper"""

        return

    def change_stage(self, stage: str):
        """Change the stage"""

        self._stage = stage

    def prepare(self):
        """Prepare the golden datas"""

        return

    def parse(self):
        """Parse the relax module"""

        return

    def redirect_forward(self, *inputs, msc_name: str = None):
        """Redirect forward method"""

        # print("[TMINFO] graph_module " + str(self._graph_module))
        # print("example_inputs " + str(self._example_inputs))
        print("inputs " + str(inputs))
        return self._graph_module.forward(*inputs)

    def optimize(self, dataset: dict = None, workspace: str = "Optimize"):
        """Optimize the model

        Parameters
        ----------
        dataset: dict
            The dataset.
        workspace: str
            The workspace.
        """

        self._managers = {}

        import torch  # type: ignore[import]
        from torch import fx  # type: ignore[import]
        from torch import _dynamo as dynamo  # type: ignore[import]

        def _optimize(graph_module: fx.GraphModule, example_inputs):
            name = "msc_" + str(len(self._managers))
            self._managers[name] = {"model": graph_module}
            return partial(self.redirect_forward, msc_name=name)

        dynamo.reset()
        self._managers = {}
        self._optimized_model = torch.compile(self._meta_model, backend=_optimize)
        assert MSCStage.PREPARE in dataset, "{} is needed to optimize model"
        self.change_stage(MSCStage.PREPARE)
        cnt, max_golden = 0, dataset[MSCStage.PREPARE].get("max_golden", 5)
        for inputs in dataset[MSCStage.PREPARE]["loader"]():
            if cnt >= max_golden > 0:
                break
            print("running {} th forward".format(cnt))
            if isinstance(inputs, (list, tuple)):
                self._optimized_model(*inputs)
            else:
                self._optimized_model(inputs)
            cnt += 1
        raise Exception("stop here!!")
        return self

    def _get_model(self) -> Any:
        return self._compiled_model or self._optimized_model or self._meta_model

    @classmethod
    def create_config(cls, **kwargs) -> dict:
        """Create config for msc pipeline

        Parameters
        ----------
        kwargs: dict
            The config kwargs.
        """

        return create_config([], [], MSCFramework.TORCH, **kwargs)

    @property
    def optimized(self):
        return self._optimized_model is not None

    @property
    def compiled(self):
        return self._compiled_model is not None

    @property
    def logger(self):
        return self._common_config["logger"]

    @classmethod
    def model_type(cls):
        return MSCFramework.MSC


class TorchDynamic(BaseDynamic):
    def optimize(self, dataset: dict = None, workspace: str = "Optimize"):
        """Optimize the model

        Parameters
        ----------
        dataset: dict
            The dataset.
        workspace: str
            The workspace.
        """

        import torch  # type: ignore[import]
        from torch import fx  # type: ignore[import]
        from torch import _dynamo as dynamo  # type: ignore[import]

        def _optimize(graph_module: fx.GraphModule, example_inputs):
            name = "msc_" + str(len(self._managers))
            self._managers[name] = {"model": graph_module}
            return partial(self.redirect_forward, msc_name=name)

        dynamo.reset()
        return torch.compile(self._meta_model, backend=_optimize)
