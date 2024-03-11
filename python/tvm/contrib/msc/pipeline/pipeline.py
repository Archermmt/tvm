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
# pylint: disable=import-outside-toplevel
"""tvm.contrib.msc.pipeline.pipeline"""

import os
import time
import json
import logging
from typing import Dict, Any, Union, List, Tuple
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
from .config import support_tool, get_tool_stage
from .worker import BasePipeWorker, MSCPipeWorker


class BasePipeline(object):
    """Base Pipeline of MSC

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    plugins: dict
        The plugins for pipeline.
    run_optimize: bool
        Whether to run optimize.
    run_compile: bool
        Whether to run compile.
    root: str
        The root path for files.
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        plugins: dict = None,
        run_optimize: bool = True,
        run_compile: bool = True,
        root: str = None,
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

        MSCMap.reset()
        self._model, self._meta_config = model, config
        self._config = msc_utils.copy_dict(config)
        self._model_type = config["model_type"]
        self._optimize_type = config.get(MSCStage.OPTIMIZE, {}).get("run_type", self._model_type)
        self._compile_type = config.get(MSCStage.COMPILE, {}).get("run_type", self._model_type)
        if not run_optimize and MSCStage.OPTIMIZE in self._config:
            self._config.pop(MSCStage.OPTIMIZE)
        if not run_compile and MSCStage.COMPILE in self._config:
            self._config.pop(MSCStage.COMPILE)
        self._plugins = load_plugins(plugins) if plugins else {}
        use_cache = config.get("use_cache", True)
        self._workspace = msc_utils.set_workspace(config.get("workspace"), use_cache)
        if "logger" in config:
            self._logger = config["logger"]
            MSCMap.set(MSCKey.GLOBALE_LOGGER, self._logger)
        else:
            verbose = config.get("verbose", "info")
            log_file = config.get("log_file") or self._workspace.relpath(
                "MSC_LOG", keep_history=False
            )
            self._logger = msc_utils.set_global_logger(verbose, log_file)
        msc_utils.time_stamp(MSCStage.SETUP)
        self._logger.info(msc_utils.msg_block(self.pipe_mark("SETUP"), self.setup()))

    def setup(self) -> dict:
        """Setup the pipeline

        Returns
        -------
        info: dict
            The setup info.
        """

        # register plugins
        if self._plugins:
            for t in [self._model_type, self._optimize_type, self._compile_type]:
                assert t in self._plugins, "Missing plugin for {}".format(t)
            for name, plugin in self._plugins[self._model_type].get_ops_info().items():
                _ffi_api.RegisterPlugin(name, msc_utils.dump_dict(plugin))
        # init report
        self._report = {
            "success": False,
            "info": {
                "workspace": self._workspace.path,
                "log_file": msc_utils.get_log_file(self._logger),
            },
            "duration": {},
            "profile": {},
        }
        return {
            "workspace": self._workspace.path,
            "log_file": msc_utils.get_log_file(self._logger),
            "plugins": self._plugins,
            "config": self._config,
        }

    def run_pipe(self) -> dict:
        """Run the pipeline and return object.

        Returns
        -------
        report:
            The pipeline report.
        """

        err_msg, err_info = None, None
        try:
            self.prepare()
            self.parse()
            if MSCStage.BASELINE in self._config:
                self.baseline()
            if MSCStage.OPTIMIZE in self._config:
                self.optimize()
            if MSCStage.COMPILE in self._config:
                self.compile()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            err_msg = "Pipeline failed: " + str(exc)
            err_info = traceback.format_exc()
        self.summary(err_msg, err_info)
        self._logger.info(msc_utils.msg_block("SUMMARY", self._report, 0))
        self._workspace.finalize()
        return self._report

    def prepare(self):
        """Prepare datas for the pipeline."""

        msc_utils.time_stamp(MSCStage.PREPARE)
        info, report = self._prepare()
        self._record_stage(MSCStage.PREPARE, info, report)

    def _prepare(self) -> Tuple[dict, dict]:
        """Prepare datas for the pipeline.

        Returns
        -------
        info: dict
            The info of prepare
        report: dict
            The report of prepare.
        """

        raise NotImplementedError("_prepare is not implemented in " + str(self.__class__))

    def parse(self):
        """Parse relax module for the pipeline."""

        msc_utils.time_stamp(MSCStage.PARSE)
        info, report = self._parse()
        self._record_stage(MSCStage.PARSE, info, report)

    def _parse(self) -> Tuple[dict, dict]:
        """Parse relax module for the pipeline.

        Returns
        -------
        info: dict
            The info of parse.
        report: dict
            The report of parse.
        """

        raise NotImplementedError("_parse is not implemented in " + str(self.__class__))

    def baseline(self) -> Tuple[dict, dict]:
        """Run the baseline.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        return self.run_stage(MSCStage.BASELINE)

    def optimize(self) -> Tuple[dict, dict]:
        """Run the optimize.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        info, report = self.run_stage(MSCStage.OPTIMIZE)
        self._optimized = True
        return info, report

    def compile(self) -> Tuple[dict, dict]:
        """Run the compile.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        info, report = self.run_stage(MSCStage.COMPILE)
        self._compiled = True
        return info, report

    def run_stage(self, stage: str) -> Tuple[dict, dict]:
        """Run the stage.

        Parameters
        ----------
        stage: str
            The compile stage.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        msc_utils.time_stamp(stage)
        applied_tools = []
        for tool in self._config.get("tools", []):
            run_type = tool.get("run_type", self._config[stage]["run_type"])
            if not support_tool(tool, stage, run_type):
                continue
            if self._tool_applied(tool["tool_type"]):
                continue
            tool_stage = get_tool_stage(tool["tool_type"])
            msc_utils.time_stamp(stage + "." + tool_stage)
            info, report = self._apply_tool(
                stage, tool["tool_type"], applied_tools, self._get_loader(tool_stage)
            )
            self._record_stage(stage + "." + tool_stage, info, report)
            if tool.get("apply_once", False):
                msg = "Ignore apply once tool " + str(tool["tool_type"])
                self._logger.debug(self.pipe_mark(msg))
            else:
                applied_tools.append(tool["tool_type"])
        info, report = self._run_stage(stage)
        self._record_stage(stage, info, report)

    def _tool_applied(self, tool_type: str) -> bool:
        """Check if the tool is applied

        Parameters
        ----------
        tool_type: str
            The tool type.

        Returns
        -------
        applied: bool
            Whether the tool is applied.
        """

        return False

    def _apply_tool(self, stage: str, tool_type: str, applied_tools: List[str], loader: callable):
        """Apply tool with runner

        Parameters
        ----------
        stage: str
            The compile stage.
        tool_type: str
            The tool type to apply.
        applied_tools: list<str>
            The applied tool types.
        loader: callable
            The data loader.

        Returns
        -------
        info: dict
            The info of apply tool.
        report: dict
            The report of apply tool.
        """

        raise NotImplementedError("_apply_tool is not implemented in " + str(self.__class__))

    def _run_stage(self, stage: str):
        """Run the stage.

        Parameters
        ----------
        stage: str
            The compile stage.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        raise NotImplementedError("_run_stage is not implemented in " + str(self.__class__))

    def summary(self, err_msg=None, err_info: str = None):
        """Summary the pipeline.

        Parameters
        ----------
        err_msg: str
            The error message.
        err_info: str
            The error info.

        Returns
        -------
        report: dict
            The report of the pipeline.
        """

        msc_utils.time_stamp(MSCStage.SUMMARY, False)
        if err_msg:
            self._report.update({"success": False, "err_msg": err_msg, "err_info": err_info})
        else:
            self._report["success"] = True
        self._report["duration"] = msc_utils.get_duration()
        return self._report

    def destory(self, keep_workspace: bool = False):
        """Destroy the pipeline

        Parameters
        ----------
        keep_workspace: bool
            Whether to keep workspace.
        """

        self._destory()
        if not keep_workspace:
            self._workspace.destory()
        msc_utils.remove_loggers()

    def _destory(self):
        """Destroy the pipeline."""

        raise NotImplementedError("_destory is not implemented in " + str(self.__class__))

    def export(self, path: str = None, dump: bool = True) -> Union[str, dict]:
        """Export the pipeline

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

        path = path or "msc_export"
        if path.endswith(".tar.gz"):
            folder, dump = msc_utils.msc_dir(path.replace(".tar.gz", ""), keep_history=False), True
        else:
            folder = msc_utils.msc_dir(path, keep_history=False)

        def _to_root_mark(val):
            if isinstance(val, str) and folder.path != val and folder.path in val:
                return val.replace(folder.path, MSCKey.ROOT_MARK)
            return val

        # export compiled
        if self._compiled:
            if not dump:
                return self._runner.runnable
            model = self._runner.export_runnable(folder)
            if self._plugins:
                plugin = self._plugins[self.compile_type]
                model["plugins"] = plugin.copy_libs(folder.create_dir("plugins"))
            model.update(
                {
                    "device": self._runner.device,
                    "model_type": self.compile_type,
                    "abstract": self._runner.model_info,
                }
            )
            # save golden
            num_golden = self._config[MSCStage.EXPORT].get("num_golden", 0)
            if num_golden > 0:
                saver_options = {
                    "input_names": [i[0] for i in self._config["inputs"]],
                    "output_names": self._config["outputs"],
                }
                batch_cnt, model["golden"] = 0, folder.create_dir("golden").path
                with msc_utils.IODataSaver(model["golden"], saver_options) as saver:
                    for inputs in self._get_loader()():
                        if batch_cnt >= num_golden:
                            break
                        batch_cnt = saver.save_batch(inputs, self._runner.run(inputs))
            model = msc_utils.map_dict(model, _to_root_mark)
            with open(folder.relpath("model.json"), "w") as f:
                f.write(json.dumps(model, indent=2))
        else:
            if dump:
                plugins = export_plugins(self._plugins, folder.create_dir("plugins"))
            else:
                plugins = self._plugins

            pipeline = {
                "model": self.export_model(folder.create_dir("model"), dump),
                "config": self.export_config(folder, dump),
                "plugins": plugins,
                "root": folder.path,
            }
            pipeline = msc_utils.map_dict(pipeline, _to_root_mark)
            if not dump:
                return pipeline
            with open(folder.relpath("pipeline.json"), "w") as f:
                f.write(json.dumps(pipeline, indent=2))
        # copy common files
        if self._optimized or self._compiled:
            stage = MSCStage.COMPILE if self._compiled else MSCStage.OPTIMIZE
            msc_utils.get_visual_dir().copy(stage, folder.relpath("visualize"))
            for log_h in self._logger.handlers:
                if isinstance(log_h, logging.FileHandler):
                    folder.copy(log_h.baseFilename)
            with open(folder.relpath("report.json"), "w") as f:
                f.write(json.dumps(self._report, indent=2))
        folder.finalize()
        if path.endswith(".tar.gz"):
            msc_utils.pack_folder(path.replace(".tar.gz", ""), "tar.gz")
        return path

    def export_model(self, folder: msc_utils.MSCDirectory, dump: bool = True) -> Any:
        """Export the model

        Parameters
        ----------
        folder: MSCDirectory
            The export folder.
        dump: bool
            Whether to dump info.

        Returns
        -------
        exported:
            The exported model.
        """

        if self._optimized:
            module = self._runner.export_module(folder)
            if not dump:
                return module
            path = folder.relpath("model.json")
            with open(path, "w") as f:
                f.write(tvm.ir.save_json(module))
            return {"model": path}
        if not dump:
            return self._model
        return self._get_runner_cls(self._model_type).dump_nativate(
            self._model, folder, **self._config[MSCStage.EXPORT]
        )

    def export_config(self, folder: msc_utils.MSCDirectory, dump: bool = True) -> dict:
        """Export the config

        Parameters
        ----------
        folder: MSCDirectory
            The export folder.
        dump: bool
            Whether to dump info.

        Returns
        -------
        config: dict
            The updated config.
        """

        # dump the dataloader
        def _save_dataset(name, info, dump: bool):
            loader, max_batch = info["loader"], info.get("max_batch", -1)
            data_folder = folder.create_dir("dataset")
            if isinstance(loader, str) and msc_utils.is_callable(loader):
                path, func_name = loader.split(":")
                exp_loader = data_folder.copy(path) + ":" + func_name
            elif msc_utils.is_io_dataset(loader):
                exp_loader = data_folder.copy(loader, name)
            elif callable(loader) and dump:
                saver_options = {
                    "input_names": [i[0] for i in self._config["inputs"]],
                    "output_names": self._config["outputs"],
                }
                batch_cnt = 0
                exp_loader = data_folder.create_dir(name).path
                with msc_utils.IODataSaver(exp_loader, saver_options) as saver:
                    for inputs in loader():
                        if batch_cnt >= max_batch > 0:
                            break
                        batch_cnt = saver.save_batch(inputs)
            else:
                exp_loader = loader
            return {"loader": exp_loader, "max_batch": max_batch}

        config = msc_utils.copy_dict(self._meta_config)
        config["dataset"] = {
            k: _save_dataset(k, v, dump) for k, v in self._config["dataset"].items()
        }
        if self._optimized:
            config["model_type"] = MSCFramework.TVM
            for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE]:
                if stage in config:
                    config.pop(stage)
            if "profile" in config[MSCStage.COMPILE]:
                config[MSCStage.COMPILE]["profile"].setdefault("check", {})["err_rate"] = -1
            config["tools"] = []
            for tool in self._config.get("tools", []):
                if not support_tool(tool, MSCStage.COMPILE, self._compile_type):
                    continue
                run_tool = self.runner.get_tool(tool["tool_type"])
                tool["tool_config"] = run_tool.export_config(tool["tool_config"], folder)
                if tool["tool_config"]:
                    config["tools"].append(tool)
                else:
                    self._logger.info(
                        "Skip compile with tool %s as no config exported", tool["tool_type"]
                    )
        # remove not serializable items
        if dump:
            remove_keys = {"workspace", "logger"}
            config = {k: v for k, v in config.items() if k not in remove_keys}
        return config

    def _get_loader(self, name: str = MSCStage.PREPARE) -> Any:
        """Get the data loader"""

        config = self._config["dataset"].get(name, self._config["dataset"][MSCStage.PREPARE])
        source_loader = config.get("loader")
        assert source_loader, "Dataset loader should be given for msc pipeline"
        if source_loader == "from_random":
            max_batch = config.get("max_batch", 5)

            def get_random():
                for _ in range(max_batch):
                    yield {i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]}

            loader, source_type = get_random, "Random"
        elif msc_utils.is_io_dataset(source_loader):
            max_batch = config.get("max_batch", -1)

            def load_datas():
                for inputs, _ in msc_utils.IODataLoader(source_loader, end=max_batch):
                    yield inputs

            loader, source_type = load_datas, "IOData"
        elif callable(source_loader):
            max_batch = config.get("max_batch", -1)
            load_kwargs = config.get("load_kwargs", {})

            def get_source():
                for idx, inputs in enumerate(source_loader(**load_kwargs)):
                    if idx >= max_batch > 0:
                        break
                    yield inputs

            loader, source_type = get_source, "Custom"
        else:
            raise TypeError(
                "Unexpected source loader {}({})".format(source_loader, type(source_loader))
            )
        msg = "Create data loader({}) {}({})".format(name, loader.__name__, source_type)
        self._logger.debug(self.pipe_mark(msg))
        return loader

    def create_worker(self, model: Any = None, **extra_config):
        """Create pipe worker

        Parameters
        -------
        model: Any
            The raw model in framwork.
        extra_config: dict
            The extra config for pipeline.

        Returns
        -------
        worker: str
            The message with mark.
        """

        config = msc_utils.update_dict(msc_utils.copy_dict(self._config), extra_config)
        return self.worker_cls(
            model or self._model, config, self._workspace, self._plugins, self._logger
        )

    def add_report(self, stage: str, report: Any):
        """Add stage report to report

        Parameters
        -------
        stage: str
            The compile stage.
        report:
            The report of the stage
        """

        self._report[stage] = report

    def _record_stage(self, stage: str, info: dict = None, report: dict = None):
        """Record the stage

        Parameters
        -------
        stage: str
            The compile stage
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        if info:
            self._logger.info(msc_utils.msg_block(self.pipe_mark(stage.upper()), info))
        if report:
            self.add_report(stage, report)

    def pipe_mark(self, msg: Any) -> str:
        """Mark the message with pipeline info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "Pipeline " + str(msg)

    @property
    def worker_cls(self):
        return BasePipeWorker

    @property
    def report(self):
        return self._report

    @property
    def model_type(self):
        return self._model_type

    @property
    def optimize_type(self):
        return self._optimize_type

    @property
    def compile_type(self):
        return self._compile_type
