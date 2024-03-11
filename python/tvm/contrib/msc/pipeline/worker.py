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
"""tvm.contrib.msc.pipeline.worker"""

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


class BasePipeWorker(object):
    """Base Worker of MSC pipeline

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    workspace: MSCDirectory
        The workspace.
    plugins: dict
        The plugins for pipeline.
    run_optimize: bool
        Whether to run optimize.
    run_compile: bool
        Whether to run compile.
    logger: logging.Logger
        The logger.
    name: str
        The name of the worker.
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        workspace: msc_utils.MSCDirectory,
        plugins: dict = None,
        logger: logging.Logger = None,
        name: str = "main",
    ):
        # check/set default stage
        for key in ["inputs", "outputs", "dataset"]:
            assert stage in config, "Missing {} in config".format(key)
        for stage in [MSCStage.PREPARE, MSCStage.PARSE, MSCStage.COMPILE, MSCStage.EXPORT]:
            config.setdefault(stage, {})

        self._config = config
        self._workspace = workspace
        self._plugins = plugins
        self._model_type = config["model_type"]
        self._optimize_type = config.get(MSCStage.OPTIMIZE, {}).get("run_type", self._model_type)
        self._compile_type = config.get(MSCStage.COMPILE, {}).get("run_type", self._model_type)
        runner_cls = self._get_runner_cls(self._model_type)
        self._model, self._device, self._training = runner_cls.load_native(model, config)
        self._verbose = config.get("verbose", "info")
        self._logger = logger or msc_utils.get_global_logger()
        self._name = name
        self._optimized, self._compiled = False, False
        self.setup()

    def setup(self) -> dict:
        """Setup the manager

        Returns
        -------
        config: dict
            The updated config.
        """

        self._debug_levels = self.update_config()
        self._tools_config = {t["tool_type"]: t for t in self._config["tools"]}
        self._relax_mod, self._sample_inputs = None, None
        self._tools_config, self._runner = [], None

    def update_config(self) -> dict:
        """Update config

        Returns
        -------
        debug_levels: dict
            The debug_levels.
        """

        debug_levels = {}
        self._config = self._get_runner_cls(self._model_type).update_config(
            MSCStage.PARSE, self._config, self._model
        )

        # update runner config
        for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]:
            if stage not in self._config:
                continue
            if "run_type" not in self._config[stage]:
                self._config[stage]["run_type"] = self._model_type
            runner_cls = self._get_runner_cls(self._config[stage]["run_type"])
            self._config = runner_cls.update_config(stage, self._config, self._model)

        # update tool config
        if self._config.get("tools"):
            self._config["tools"] = self._update_tools_config(self._config["tools"])

        # update export config
        self._config[MSCStage.EXPORT].update(
            {"inputs": self._config["inputs"], "outputs": self._config["outputs"]}
        )

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
            if stage not in self._config:
                continue
            debug_levels = _set_debug_level(stage, self._config[stage]["run_config"], debug_level)
            for t_config in self._config.get("tools", []):
                if not support_tool(t_config, stage, self._config[stage]["run_type"]):
                    continue
                t_stage = stage + "." + get_tool_stage(t_config["tool_type"])
                debug_levels = _set_debug_level(t_stage, t_config["tool_config"], debug_level)
        ordered_keys = [
            "model_type",
            "inputs",
            "outputs",
            "dataset",
            "tools",
            MSCStage.PREPARE,
            MSCStage.PARSE,
            MSCStage.BASELINE,
            MSCStage.OPTIMIZE,
            MSCStage.COMPILE,
            MSCStage.EXPORT,
        ]
        self._config = {k: self._config[k] for k in ordered_keys if k in self._config}
        return debug_levels

    def _update_tools_config(self, tools: List[dict]) -> List[dict]:
        """Update tool in stage config.

        Parameters
        ----------
        tools: list<dict>
            The config of tools.

        Returns
        -------
        tools: list<dict>
            The updated config of tools.
        """

        for tool in tools:
            tool_config = tool["tool_config"]
            if "plan_file" not in tool_config:
                tool_config["plan_file"] = "msc_{}.json".format(tool["tool_type"])
            tool_config["plan_file"] = msc_utils.to_abs_path(
                tool_config["plan_file"], msc_utils.get_config_dir()
            )
        return tools

    def prepare(self) -> Tuple[dict, dict]:
        """Prepare datas for the pipeline.

        Returns
        -------
        info: dict
            The info of prepare.
        report: dict
            The report of prepare.
        """

        stage_config = self._config[MSCStage.PREPARE]
        use_cache = self._config.get("use_cache", True)
        runner_cls = self._get_runner_cls(self._model_type)
        run_func = runner_cls.run_native if hasattr(runner_cls, "run_native") else None
        input_names = [i[0] for i in self._config["inputs"]]

        # create golden
        if "golden" in self._config["dataset"]:
            golden_folder = self._config["dataset"]["golden"]["loader"]
        else:
            golden_folder = msc_utils.get_dataset_dir().relpath("Golden", use_cache)
        if msc_utils.is_io_dataset(golden_folder):
            loader, source_type = msc_utils.IODataLoader(golden_folder), "Cache"
            self._sample_inputs = loader[0][0]
            datas_info = loader.info
            msg = "Load {} golden from {}".format(len(loader), golden_folder)
            self._logger.debug(self.worker_mark(msg))
        elif run_func:
            loader, source_type = self._get_loader(MSCStage.PREPARE), "Native"
            saver_options = {"input_names": input_names, "output_names": self._config["outputs"]}
            cnt, max_golden = 0, self._config["dataset"][MSCStage.PREPARE].get("max_golden", 5)
            with msc_utils.IODataSaver(golden_folder, saver_options) as saver:
                for inputs in loader():
                    if cnt >= max_golden > 0:
                        break
                    if not self._sample_inputs:
                        self._sample_inputs = {
                            k: msc_utils.cast_array(v) for k, v in inputs.items()
                        }
                    outputs, _ = run_func(self._model, inputs, input_names, self._config["outputs"])
                    cnt = saver.save_batch(inputs, outputs)
                datas_info = saver.info
            msg = "Saved {} golden to {}".format(cnt, golden_folder)
            self._logger.debug(self.worker_mark(msg))
        else:
            raise Exception("golden_folder or runner should given to save golden")
        self._config["dataset"]["golden"] = {"loader": golden_folder, "max_batch": -1}

        def _to_abstract(info: dict) -> dict:
            def _to_tensor_str(info):
                return "{},{}".format(";".join([str(s) for s in info["shape"]]), info["dtype"])

            return {
                "num_datas": info["num_datas"],
                "inputs": {n: _to_tensor_str(i) for n, i in info["inputs"].items()},
                "outputs": {n: _to_tensor_str(o) for n, o in info["outputs"].items()},
            }

        info = {
            "golden_folder({})".format(source_type): golden_folder,
            "datas_info": _to_abstract(datas_info),
            "smaple_inputs": self._sample_inputs,
        }

        # profile
        report = {}
        if "profile" in stage_config and run_func:
            benchmark = stage_config["profile"].get("benchmark", {})
            benchmark["repeat"] = self._get_repeat(benchmark)
            pre_msg = "Prepare profile with {}({})".format(run_func.__name__, benchmark)
            self._logger.debug(self.worker_mark(pre_msg))
            _, avg_time = run_func(
                self._model, self._sample_inputs, input_names, self._config["outputs"], **benchmark
            )
            report = {"latency": "{:.2f} ms @ {}".format(avg_time, self._device)}
            msg = "profile(prepare) {} times -> {}".format(benchmark["repeat"], info["latency"])
            self._logger.info(self.worker_mark(msg))

        return info, report

    def parse(self) -> Tuple[dict, dict]:
        """Parse the model to IRModule.

        Returns
        -------
        info: dict
            The info of parse.
        report: dict
            The report of parse.
        """

        stage_config = self._config[MSCStage.PARSE]
        if self._config.get("use_cache", True):
            cache_path = (
                msc_utils.get_cache_dir().create_dir(MSCStage.PARSE).relpath("parsed_relax.json")
            )
        else:
            cache_path = None
        info = {}
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path, "r") as f:
                self._relax_mod = tvm.ir.load_json(f.read())
            info["cache"] = cache_path
        else:
            info = {"parser": stage_config["parser"], "config": stage_config.get("parse_config")}
            parse_config = msc_utils.copy_dict(stage_config.get("parse_config", {}))
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
                    msg = "Transform for {}({})".format(run_type, stage)
                    self._logger.info(self.worker_mark(msg))
                    self._relax_mod = runner_cls.target_transform(self._relax_mod)
            if cache_path:
                with open(cache_path, "w") as f:
                    f.write(tvm.ir.save_json(self._relax_mod))
                msg = "Save parsed mod to " + cache_path
                self._logger.debug(self.worker_mark(msg))
        return info, {}

    def tool_applied(self, tool_type: str) -> bool:
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

        assert tool_type in self._tools_config, "Can not find tool_type " + str(tool_type)
        plan_file = self._tools_config[tool_type]["plan_file"]
        if os.path.isfile(plan_file):
            msg = "Skip {} with plan {}".format(tool_type, plan_file)
            self._logger.info(self.worker_mark(msg))
            return True
        return False

    def create_runner(
        self,
        stage: str,
        stage_config: dict = None,
        tools: List[str] = None,
        visualize: bool = True,
        profile: bool = True,
        use_cache: bool = True,
    ) -> Tuple[dict, dict]:
        """Create runner.

        Parameters
        ----------
        stage: str
            The stage name
        stage_config: dict
            The config of this stage.
        tools: list<str>
            The tools to apply.
        visualize: bool
            Whether to visualize the runner
        profile: bool
            Whether to profile the runner.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        info: dict
            The info of create runner.
        report: dict
            The report of create runner.
        """

        if self._runner:
            self._runner.destory()
        stage_config = stage_config or self._config[stage]
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
            tools_config=[self._tools_config[t] for t in tools],
            plugin=self._plugins.get(stage_config["run_type"]),
            stage=stage,
            name=self._name,
            logger=self._logger,
            **run_config,
        )
        runner.build(cache_dir=cache_dir)
        info, report = {}, {"build": "{} on {}".format(runner.framework, runner.device)}
        if visualize:
            runner.visualize(msc_utils.get_visual_dir().create_dir(stage.split(".")[0]))
        if profile and "profile" in stage_config:
            info["profile"], report["profile"] = self._profile_runner(runner, stage_config)
        if use_cache:
            runner.save_cache(cache_dir)
        return info, report

    def create_runner_with_tools(
        self, stage: str, tool_type: str, applied_tools: List[str]
    ) -> Tuple[dict, dict]:
        """Create runner with tools.

        Parameters
        ----------
        stage: str
            The stage name
        tool_type: str
            The tool type to apply.
        applied_tools: list<str>
            The applied tool types.

        Returns
        -------
        info: dict
            The info of create runner.
        report: dict
            The report of create runner.
        """

        assert tool_type in self._tools_config, "Can not find tool_type " + str(tool_type)
        tool_config = self._tools_config[tool_type]
        stage_config = {
            "run_type": tool_config.get("run_type", self._config[stage]["run_type"]),
            "run_config": self._config[stage]["run_config"],
        }
        tool_stage = get_tool_stage(tool_type)
        return self.create_runner(
            stage + "." + tool_stage,
            stage_config,
            applied_tools + [tool_type],
            visualize=False,
            profile=False,
            use_cache=False,
        )

    def _profile_runner(self, runner: BaseRunner, stage_config: str) -> dict:
        """Profile the runner.

        Parameters
        ----------
        runner: BaseRunner
            The runner to be profiled
        stage_config: dict
            The config of this stage.

        Returns
        -------
        info: dict
            The info of profile.
        report: dict
            The report of profile.
        """

        stage = runner.stage
        msc_utils.time_stamp(stage + ".profile", False)
        profile_config = stage_config["profile"]
        msg = "profile({})".format(stage)
        info, report = {}, {}

        # check accuracy
        check_config = profile_config.get("check", {})
        if check_config:
            loader = msc_utils.IODataLoader(self._config["dataset"]["golden"]["loader"])
            acc_info = {"config": check_config, "passed": 0}
            total, passed = 0, 0
            for inputs, outputs in loader:
                results = runner.run(inputs)
                iter_info = msc_utils.compare_arrays(
                    outputs,
                    results,
                    atol=check_config.get("atol", 1e-2),
                    rtol=check_config.get("rtol", 1e-2),
                )
                total += iter_info["total"]
                passed += iter_info["passed"]
                acc_info["iters"].append(iter_info["info"])
            pass_rate = float(passed) / total
            acc_info["passed"] = "{}/{}".format(passed, total)
            report["accuracy"] = "{}/{}({:.2f}%)".format(passed, total, pass_rate * 100)
            info["accuracy"] = acc_info
            msg += " test {} iters -> {}".format(len(loader), report["accuracy"])
            if runner.get_tool(ToolType.PRUNER) or runner.get_tool(ToolType.QUANTIZER):
                msg = "Disable accuracy check({}) by tools".format(stage)
                self._logger.debug(self.worker_mark(msg))
            else:
                required_err, err_rate = check_config.get("err_rate", 0), (1 - pass_rate)
                if err_rate > required_err >= 0:
                    raise Exception(
                        "Failed to profile the runner({}), err_rate {} > required {}".format(
                            stage, err_rate, required_err
                        )
                    )

        # benchmark model
        if runner.get_tool(ToolType.TRACKER):
            benchmark_config = None
            msg = "Disable benchmark ({}) by tracker".format(stage)
            self._logger.debug(self.worker_mark(msg))
        else:
            benchmark_config = profile_config.get("benchmark", {})
        if benchmark_config:
            for _ in range(benchmark_config.get("warm_up", 10)):
                runner.run(self._sample_inputs)
            start = time.time()
            repeat = self._get_repeat(benchmark_config, runner.device)
            for _ in range(repeat):
                runner.run(self._sample_inputs)
            avg_time = (time.time() - start) * 1000 / repeat
            report["latency"] = "{:.2f} ms @ {}".format(avg_time, runner.device)
            msg += " latency {} times -> {}".format(repeat, report["latency"])
        self._logger.info(self.worker_mark(msg))
        return info, report

    def get_runnable(self, ret_type: str = "runner") -> Any:
        """Return object by type.

        Parameters
        ----------
        ret_type: str
            The return type runner| model.

        Returns
        -------
        runnable:
            The runner or model.
        """

        assert self._runner, "Failed to create runner, call run_pipe first"
        if ret_type == "runner":
            return self._runner
        elif ret_type == "runnable":
            return self._runner.runnable
        elif ret_type == "model":
            return self._runner.model
        raise Exception("Unexpect return type " + str(ret_type))

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

        raise NotImplementedError("_get_runner_cls is not implemented in " + str(self.__class__))

    def destory(self):
        """Destroy the worker"""

        if self._runner:
            self._runner.destory()

    def worker_mark(self, msg: Any) -> str:
        """Mark the message with worker info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "Worker[{}]: {}".format(self._name, msg)

    def apply_tool(self, tool: dict, stage: str, applied_tools: List[str]) -> str:
        """Apply tool with runner

        Parameters
        ----------
        tool: dict
            The tool config.
        stage: str
            The compile stage.

        Returns
        -------
        plan_file: str
            The plan_file path.
        """

        self._tools_config.append(tool)
        tool_type, tool_config = tool["tool_type"], tool["tool_config"]
        tool_stage = self._get_tool_stage(tool_type)
        plan_file = tool_config["plan_file"]
        if os.path.isfile(plan_file):
            self._logger.info("Skip %s with plan %s", tool_type, plan_file)
            return plan_file
        t_stage = stage + "." + tool_stage
        msc_utils.time_stamp(t_stage)
        stage_config = {
            "run_type": tool.get("run_type", self._config[stage]["run_type"]),
            "run_config": self._config[stage]["run_config"],
        }
        runner = self._create_runner(
            t_stage, stage_config, visualize=False, profile=False, use_cache=False
        )
        if "gym_configs" in tool:
            knowledge = None
            for idx, config in enumerate(tool["gym_configs"]):
                knowledge_file = msc_utils.get_config_dir().relpath(
                    "gym_knowledge_{}.json".format(idx)
                )
                gym_mark = "GYM[{}/{}]({} @ {}) ".format(
                    idx, len(tool["gym_configs"]), runner.framework, t_stage
                )
                if os.path.isfile(knowledge_file):
                    knowledge = knowledge_file
                    self._logger.info("%sLoad from %d", gym_mark, knowledge)
                else:
                    msc_utils.time_stamp(t_stage + ".gym_{}".format(idx))
                    self._logger.info("%sStart search", gym_mark)
                    extra_config = {
                        "env": {
                            "runner": runner,
                            "data_loader": self._get_loader(tool_stage),
                            "knowledge": knowledge,
                        },
                        "verbose": self._verbose,
                    }
                    controller = create_controller(tool_stage, config, extra_config)
                    knowledge = controller.run()
                    msc_utils.save_dict(knowledge, knowledge_file)
            plan = msc_utils.load_dict(knowledge)
            self._logger.info("%sFound %d plan", gym_mark, len(plan))
            return msc_utils.save_dict(plan, plan_file)
        msc_utils.time_stamp(t_stage + ".make_plan", False)
        plan_file = runner.make_plan(tool_type, self._get_loader(tool_stage))
        if tool.get("visualize", False):
            runner.visualize(msc_utils.get_visual_dir().create_dir(stage))
        return plan_file

    @property
    def runner(self):
        return self._runner

    @property
    def model_type(self):
        return self._model_type

    @property
    def optimize_type(self):
        return self._optimize_type

    @property
    def compile_type(self):
        return self._compile_type


class MSCPipeWorker(BasePipeWorker):
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
