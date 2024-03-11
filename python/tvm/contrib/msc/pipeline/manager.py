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
"""tvm.contrib.msc.pipeline.manager"""

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
from tvm.contrib.msc.plugin.utils import export_plugins, load_plugins
from .config import support_tool, get_tool_stage
from .pipeline import MSCPipeWorker, BasePipeline


class MSCManager(BasePipeline):
    """Manager of Pipeline, process static model"""

    def setup(self) -> dict:
        """Setup the pipeline

        Returns
        -------
        info: dict
            The setup info.
        """

        self._worker = self.create_worker()
        self._config = self._worker._config
        return super().setup()

    def _prepare(self) -> Tuple[dict, dict]:
        """Prepare datas for the pipeline.

        Returns
        -------
        report: dict
            The report of prepare.
        info: dict
            The info of prepare
        """

        return self._worker.prepare()

    def _parse(self) -> Tuple[dict, dict]:
        """Parse relax module for the pipeline.

        Returns
        -------
        info: dict
            The info of parse.
        report: dict
            The report of parse.
        """

        return self._worker.parse()

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

        return self._worker.tool_applied(tool_type)

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

        tool_stage = get_tool_stage(tool_type)
        t_stage = stage + "." + tool_stage
        stage_config = {
            "run_type": tool.get("run_type", self._config[stage]["run_type"]),
            "run_config": self._config[stage]["run_config"],
        }
        runner = self._worker.create_runner(
            t_stage, stage_config, visualize=False, profile=False, use_cache=False
        )
        if "gym_configs" in tool:
            knowledge = None
            for idx, config in enumerate(tool["gym_configs"]):
                knowledge_file = msc_utils.get_config_dir().relpath(
                    "gym_knowledge_{}.json".format(idx)
                )
                gym_mark = "GYM[{}/{}]({} @ {}) ".format(
                    idx, len(tool["gym_configs"]), self._config[stage]["run_type"], tool_stage
                )
                if os.path.isfile(knowledge_file):
                    knowledge = knowledge_file
                    msg = "{}load from {}".format(gym_mark, knowledge)
                    self._logger.info(self.pipe_mark(msg))
                else:
                    msc_utils.time_stamp(tool_stage + ".gym_{}".format(idx))
                    self._logger.info(self.pipe_mark(gym_mark + "start search"))
                    extra_config = {
                        "env": {
                            "runner": runner,
                            "data_loader": loader,
                            "knowledge": knowledge,
                        },
                        "verbose": self._verbose,
                    }
                    controller = create_controller(tool_stage, config, extra_config)
                    knowledge = controller.run()
                    msc_utils.save_dict(knowledge, knowledge_file)
            plan = msc_utils.load_dict(knowledge)
            msg = "gym found {} plan".format(len(plan))
            self._logger.info(self.pipe_mark(msg))
            msc_utils.save_dict(plan, plan_file)
        msc_utils.time_stamp(t_stage + ".make_plan", False)
        plan_file = runner.make_plan(tool_type, loader)
        if tool.get("visualize", False):
            runner.visualize(msc_utils.get_visual_dir().create_dir(stage.split(".")[0]))
        return plan_file

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

        return self._worker.create_runner(stage)

    def _destory(self):
        """Destory the pipeline"""

        self._worker.destory()

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

        return "Manager " + str(msg)

    @property
    def worker_cls(self):
        return MSCPipeWorker

    def parse(self) -> tvm.IRModule:
        """Parse the model to IRModule.

        Returns
        -------
        relax_mod: tvm.IRModule
            The parsed module.
        """

        msc_utils.time_stamp(MSCStage.PARSE)
        stage_config = self._config[MSCStage.PARSE]
        if self._config.get("use_cache", True):
            cache_path = (
                msc_utils.get_cache_dir().create_dir(MSCStage.PARSE).relpath("parsed_relax.json")
            )
        else:
            cache_path = None
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path, "r") as f:
                self._relax_mod = tvm.ir.load_json(f.read())
            self._logger.info("Load parsed mod from %s", cache_path)
        else:
            parse_config = msc_utils.copy_dict(stage_config.get("parse_config", {}))
            parse_info = {"parser": stage_config["parser"], "config": parse_config}
            self._logger.info(msc_utils.msg_block("PARSE", parse_info))
            parse_config["as_msc"] = False
            if self._model_type in self._plugins:
                plugin = self._plugins[self._model_type]
                parse_config["custom_convert_map"] = plugin.get_convert_map()
            self._relax_mod, _ = stage_config["parser"](self._model, **parse_config)
            transformed = set()
            for stage in [MSCStage.OPTIMIZE, MSCStage.COMPILE]:
                if stage not in self._config:
                    continue
                run_type = self._config[stage]["run_type"]
                if run_type in transformed:
                    continue
                transformed.add(run_type)
                runner_cls = self._get_runner_cls(run_type)
                if hasattr(runner_cls, "target_transform"):
                    self._logger.info("Transform for %s(%s)", run_type, stage)
                    self._relax_mod = runner_cls.target_transform(self._relax_mod)
            if cache_path:
                with open(cache_path, "w") as f:
                    f.write(tvm.ir.save_json(self._relax_mod))
                self._logger.debug("Save parsed mod to %s", cache_path)
        return self._relax_mod

    def _run_stage(self, stage: str) -> BaseRunner:
        """Run the stage.

        Parameters
        ----------
        stage: str
            The compile stage.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        msc_utils.time_stamp(stage)
        self.apply_tools(stage)
        self._runner = self._create_runner(
            stage,
            self._config[stage],
            use_cache=self._config.get("use_cache", True),
        )
        return self._runner

    def baseline(self) -> BaseRunner:
        """Run the baseline.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        return self._run_stage(MSCStage.BASELINE)

    def optimize(self) -> BaseRunner:
        """Run the optimize and return object.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        runner = self._run_stage(MSCStage.OPTIMIZE)
        self._optimized = True
        return runner

    def compile(self) -> BaseRunner:
        """Run the compile and return object.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        runner = self._run_stage(MSCStage.COMPILE)
        self._compiled = True
        return runner

    def apply_tools(self, stage: str):
        """Apply tools for a stage.

        Parameters
        ----------
        stage: str
            The compile stage.
        """

        self._tools_config = []
        for tool in self._config.get("tools", []):
            run_type = tool.get("run_type", self._config[stage]["run_type"])
            if not support_tool(tool, stage, run_type):
                continue
            self._apply_tool(tool, stage)
            if tool.get("apply_once", False):
                self._logger.debug("Remove apply once tool %s", tool["tool_type"])
                self._tools_config = self._tools_config[:-1]

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
            log_file = msc_utils.get_log_file(self._logger)
            if log_file:
                folder.copy(log_file)
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

    def destory(self, keep_workspace: bool = False):
        """Destroy the manager

        Parameters
        ----------
        keep_workspace: bool
            Whether to keep workspace.
        """

        if self._runner:
            self._runner.destory()
        if not keep_workspace:
            self._workspace.destory()
        msc_utils.remove_loggers()

    def _create_runner(
        self,
        stage: str,
        stage_config: dict,
        visualize: bool = True,
        profile: bool = True,
        use_cache: bool = True,
    ) -> BaseRunner:
        """Create runner.

        Parameters
        ----------
        stage: str
            The stage name
        stage_config: dict
            The config of this stage.
        visualize: bool
            Whether to visualize the runner
        profile: bool
            Whether to profile the runner.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        if self._runner:
            self._runner.destory()
        cache_dir = msc_utils.get_cache_dir().create_dir(stage) if use_cache else None
        msc_utils.time_stamp(stage + ".build", False)
        runner_cls = self._get_runner_cls(stage_config["run_type"])
        run_config = msc_utils.copy_dict(stage_config.get("run_config"))
        if "generate_config" not in run_config:
            run_config["generate_config"] = {}
        cleanup = self._debug_levels.get(stage, 0) == 0
        run_config["generate_config"]["build_folder"] = msc_utils.get_build_dir().create_dir(
            stage, cleanup=cleanup
        )
        if "device" not in run_config:
            run_config["device"] = self._device
        if "training" not in run_config:
            run_config["training"] = self._training
        # Build runner
        runner = runner_cls(
            self._relax_mod,
            tools_config=self._tools_config,
            plugin=self._plugins.get(stage_config["run_type"]),
            stage=stage,
            logger=self._logger,
            **run_config,
        )
        runner.build(cache_dir=cache_dir)
        self._report["info"][stage + "_type"] = "{}({})".format(runner.framework, runner.device)
        if visualize:
            runner.visualize(msc_utils.get_visual_dir().create_dir(stage.split(".")[0]))
        if profile and "profile" in stage_config:
            self._report["profile"][stage] = self._profile_runner(runner, stage_config)
        if use_cache:
            runner.save_cache(cache_dir)
        return runner

    def _get_repeat(self, benchmark: dict, device: str = None) -> int:
        """Get the repeat number for benchmark

        Parameters
        ----------
        benchmark: dict
            The benchmark config.
        device: str
            The device name

        Returns
        -------
        repeat: int
            The repeat number.
        """

        device = device or self._device
        repeat = benchmark.get("repeat", -1)
        if repeat == -1:
            repeat = 500 if device.startswith("cuda") else 10
        return repeat

    @property
    def runner(self):
        return self._runner

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


class MSCManager(BaseManager):
    """Normal manager in MSC"""

    def _get_runner_cls(self, run_type: str) -> BaseRunner:
        """Get the runner cls by type

        Parameters
        ----------
        run_type: str
            The run type.

        Returns
        -------
        runner_cls: class
            The runner class.
        """

        if run_type == MSCFramework.TVM:
            from tvm.contrib.msc.framework.tvm.runtime import TVMRunner

            runner_cls = TVMRunner
        elif run_type == MSCFramework.TORCH:
            from tvm.contrib.msc.framework.torch.runtime import TorchRunner

            runner_cls = TorchRunner
        elif run_type == MSCFramework.TENSORFLOW:
            from tvm.contrib.msc.framework.tensorflow.runtime import TensorflowRunner

            runner_cls = TensorflowRunner
        elif run_type == MSCFramework.TENSORRT:
            from tvm.contrib.msc.framework.tensorrt.runtime import TensorRTRunner

            runner_cls = TensorRTRunner
        else:
            raise Exception("Unexpect run_type " + str(run_type))
        return runner_cls
