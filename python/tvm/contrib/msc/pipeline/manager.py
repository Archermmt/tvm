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
"""tvm.contrib.msc.pipeline.manager"""

import os
import time
from typing import Dict, Any
import numpy as np
import traceback
import torch

import tvm
from tvm.contrib.msc.core.runtime import BaseRunner
from tvm.contrib.msc.core.utils.namespace import MSCFramework, MSCMap, MSCKey
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tvm.runtime import TVMRunner
from tvm.contrib.msc.framework.torch.frontend import from_torch
from tvm.contrib.msc.framework.torch.runtime import TorchRunner
from tvm.contrib.msc.framework.tensorflow import tf_v1
from tvm.contrib.msc.framework.tensorflow.frontend import from_tensorflow
from tvm.contrib.msc.framework.tensorflow.runtime import TensorflowRunner
from tvm.contrib.msc.framework.tensorrt.runtime import TensorRTRunner


class BaseManager(object):
    """Base Manager of MSC

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    """

    def __init__(self, model, config):
        # check config
        for stage in ["inputs", "outputs", "dataset", "prepare", "compile"]:
            assert stage in config, "{} should be given to run the pipeline".format(stage)
        self._workspace = msc_utils.set_workspace(config.get("workspace"))
        log_path = config.get("log_path") or self._workspace.relpath("MSC_LOG", keep_history=False)
        self._logger = msc_utils.set_global_logger(config.get("verbose", "info"), log_path)
        msc_utils.time_stamp("init", True)
        self._model = model
        config["workspace"] = self._workspace.path
        self._logger.info(msc_utils.msg_block("CONFIG", config))
        self._config = self.update_config(model, config)
        self._logger.debug(msc_utils.msg_block("FULL_CONFIG", self._config))
        self._relax_mod, self._runner = None, None
        self._sample_inputs = None
        self._report = {
            "success": False,
            "info": {
                "workspace": self._workspace.path,
                "log": log_path,
                "model_type": self._config["model_type"],
            },
            "duration": {},
            "profile": {},
        }

    def update_config(self, model: Any, config: dict) -> dict:
        """Update config

        Parameters
        ----------
        model: Any
            The raw model in framwork.
        config: dict
            The config for pipeline.

        Returns
        -------
        config: dict
            The updated config.
        """

        if config.get("use_cache", True):
            if "parse" in config:
                config["parse"]["cache_path"] = msc_utils.get_cache_dir().relpath(
                    "parsed_relax.json"
                )
            for stage in ["baseline", "optimize", "compile"]:
                if stage in config:
                    config[stage]["cache_dir"] = msc_utils.get_cache_dir().create_dir(stage)
        return config

    def run_pipe(self) -> dict:
        """Run the pipeline and return object.

        Returns
        -------
        summary:
            The pipeline summary.
        """

        err_msg = None
        use_cache = self._config.get("use_cache", True)
        try:
            msc_utils.time_stamp("prepare", True)
            self._sample_inputs = self.prepare(self._config["prepare"], use_cache)
            msc_utils.time_stamp("parse", True)
            self._relax_mod = self.parse(self._config["parse"], use_cache)
            if "baseline" in self._config:
                msc_utils.time_stamp("baseline", True)
                self.baseline(self._config["baseline"], use_cache)
            if "optimize" in self._config:
                msc_utils.time_stamp("optimize", True)
                self.optimize(self._config["optimize"], use_cache)
            msc_utils.time_stamp("compile", True)
            self.compile(self._config["compile"], use_cache)
            msc_utils.time_stamp("end", True, False)
        except Exception as e:
            err_msg = "Pipeline failed:{}\nTrace: {}".format(e, traceback.format_exc())
        report = self.summary(err_msg)
        self._logger.info(msc_utils.msg_block("SUMMARY", report))
        return report

    def prepare(self, stage_config: dict, use_cache: bool = False) -> Dict[str, np.ndarray]:
        """Prepare datas for the pipeline.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        sample_inputs: dict<str,np.ndarray>
            The sample inputs.
        """

        golden_folder = msc_utils.get_dataset_dir().relpath("Golden", use_cache)
        input_names = [i[0] for i in self._config["inputs"]]
        sample_inputs = None
        report = {"golden_folder": golden_folder}
        if use_cache and msc_utils.is_dataset(golden_folder):
            loader = msc_utils.MSCDataLoader(golden_folder)
            report["datas_info"] = loader.info
            sample_inputs = loader[0][0]
            self._logger.debug("Load {} cached golden from {}".format(len(loader), golden_folder))
        else:
            # create loader
            loader = self._config["dataset"].get("loader")
            assert loader, "Dataset loader should be given for msc pipeline"
            if loader.startswith("from_random"):

                def get_inputs(max_num=5):
                    for _ in range(max_num):
                        yield {
                            i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]
                        }

                data_loader = get_inputs
            elif msc_utils.is_dataset(loader):

                def get_inputs(max_num=-1):
                    for _ in range(max_num):
                        yield {
                            i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]
                        }

                data_loader = get_inputs
            else:
                data_loader = loader
            assert callable(data_loader), "Loader {} is not callable".format(data_loader)

            # save golden
            golden_cnt, max_num = 0, self._config["dataset"].get("max_num", 5)
            if "runner" in stage_config:
                with msc_utils.MSCDataSaver(
                    golden_folder, input_names, self._config["outputs"]
                ) as saver:
                    for inputs in data_loader():
                        if golden_cnt >= max_num:
                            break
                        if not sample_inputs:
                            sample_inputs = inputs
                        outputs, _ = stage_config["runner"](
                            self._model, inputs, input_names, self._config["outputs"]
                        )
                        golden_cnt = saver.save(inputs, outputs)
                    report["datas_info"] = saver.info
            elif msc_utils.is_dataset(loader):
                with msc_utils.MSCDataSaver(
                    golden_folder, input_names, self._config["outputs"]
                ) as saver:
                    for inputs, outputs in msc_utils.MSCDataLoader(loader):
                        if golden_cnt >= max_num:
                            break
                        if not sample_inputs:
                            sample_inputs = inputs
                        golden_cnt = saver.save(inputs, outputs)
                    report["datas_info"] = saver.info
            else:
                raise Exception("golden or runner should given in prepare to save golden")
            self._logger.debug("Saved {} golden to {}".format(golden_cnt, golden_folder))
        report["sample_inputs"] = sample_inputs
        self._logger.info(msc_utils.msg_block("GOLDEN", report))

        # profile
        if "profile" in stage_config and "runner" in stage_config:
            benchmark = stage_config["profile"].get("benchmark", {})
            repeat = benchmark.get("repeat", 100)
            self._logger.debug(
                "Prepare profile with {}({})".format(stage_config["runner"], benchmark)
            )
            _, avg_time = stage_config["runner"](
                self._model, sample_inputs, input_names, self._config["outputs"], **benchmark
            )
            self._logger.info("Profile(prepare) {} times -> {:.2f} ms".format(repeat, avg_time))
            self._report["profile"]["prepare"] = {"latency": "{:.2f} ms".format(avg_time)}
        return sample_inputs

    def parse(self, stage_config: dict, use_cache: bool = False) -> tvm.IRModule:
        """Parse the model to IRModule.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        relax_mod: tvm.IRModule
            The parsed module.
        """

        cache_path = stage_config.get("cache_path") if use_cache else None
        if cache_path and os.path.isfile(cache_path):
            relax_mod = tvm.ir.load_json(cache_path)
            self._logger.debug("Load parsed mod from {}".format(cache_path))
        else:
            parse_config = stage_config.get("parse_config", {})
            self._logger.debug("Parse by {}({})".format(stage_config["parser"], parse_config))
            relax_mod, _ = stage_config["parser"](self._model, as_msc=False, **parse_config)
            if cache_path:
                with open(cache_path, "w") as f:
                    f.write(tvm.ir.save_json(relax_mod))
                self._logger.debug("Save parsed mod to {}".format(cache_path))
        return relax_mod

    def baseline(self, stage_config: dict, use_cache: bool = False):
        """Run the baseline.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.
        """

        self._runner = self._create_runner(stage_config, use_cache)
        if "profile" in stage_config:
            self._report["profile"]["baseline"] = self._profile(stage_config)
        if use_cache:
            self._runner.save_cache(stage_config["cache_dir"])

    def optimize(
        self, stage_config: dict, use_cache: bool = False, ret_type: str = "runnable"
    ) -> Any:
        """Run the optimize and return object.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.
        ret_type: str
            The return type runner| model.

        Returns
        -------
        runnable:
            The runner or model.
        """

        self._runner = self._create_runner(stage_config, use_cache)
        if "profile" in stage_config:
            self._report["profile"]["optimize"] = self._profile(stage_config)
        if use_cache:
            self._runner.save_cache(stage_config["cache_dir"])
        return self.get_runnable(ret_type)

    def compile(
        self, stage_config: dict, use_cache: bool = False, ret_type: str = "runnable"
    ) -> Any:
        """Run the compile and return object.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.
        ret_type: str
            The return type runner| model.

        Returns
        -------
        runnable:
            The runner or model.
        """

        self._runner = self._create_runner(stage_config, use_cache)
        if "profile" in stage_config:
            self._report["profile"]["compile"] = self._profile(stage_config)
        if use_cache:
            self._runner.save_cache(stage_config["cache_dir"])
        return self.get_runnable(ret_type)

    def summary(self, err_msg=None):
        """Summary the pipeline.

        Parameters
        ----------
        err_msg: str
            The error message.

        Returns
        -------
        report: dict
            The report of the pipeline.
        """

        if err_msg:
            self._report.update({"success": False, "err_msg": err_msg})
        else:
            self._report["success"] = True
        self._report["duration"] = msc_utils.get_duration()
        return self._report

    def destory(self):
        """Destroy the manager"""

        if self._runner:
            self._runner.destory()
        MSCMap.delete(MSCKey.TIME_STAMPS)
        self._workspace.destory()

    def _profile(self, stage_config: str) -> dict:
        """Profile the runner.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.

        Returns
        -------
        report: dict
            The profile report.
        """

        stage = msc_utils.current_stage()
        msc_utils.time_stamp(stage + ".profile", False)
        profile_config = stage_config["profile"]
        report = {}

        # check result
        check_config = profile_config.get("check", {})
        if check_config:
            loader = msc_utils.MSCDataLoader(msc_utils.get_dataset_dir().relpath("Golden"))
            total, passed = 0, 0
            acc_report = {}
            for idx, (inputs, outputs) in enumerate(loader):
                results = self._runner.run(inputs)
                iter_report = msc_utils.compare_arrays(outputs, results)
                total += iter_report["total"]
                passed += iter_report["passed"]
                acc_report["iter_" + str(idx)] = iter_report["info"]
            title = "Check({}) pass {}/{}".format(stage, passed, total)
            self._logger.info(msc_utils.msg_block(title, acc_report))
            report["accuracy"] = "{}/{}({:.2f}%)".format(passed, total, float(passed) * 100 / total)

        # benchmark model
        benchmark_config = profile_config.get("benchmark", {})
        if benchmark_config:
            for _ in range(benchmark_config.get("warm_up", 10)):
                self._runner.run(self._sample_inputs)
            start = time.time()
            repeat = benchmark_config.get("repeat", 100)
            for _ in range(repeat):
                self._runner.run(self._sample_inputs)
            avg_time = (time.time() - start) * 1000 / repeat
            self._logger.info(
                "Profile({}) {} times on {} -> {:.2f} ms".format(
                    stage, repeat, self._runner.device, avg_time
                )
            )
            report["latency"] = "{:.2f} ms".format(avg_time)
        return report

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

        if ret_type == "runner":
            return self._runner
        elif ret_type == "runnable":
            return self._runner.runnable
        elif ret_type == "model":
            return self._runner.model
        raise Exception("Unexpect return type " + str(ret_type))

    def _create_runner(self, stage_config: dict, use_cache: bool = False) -> BaseRunner:
        """Create runner.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        stage = msc_utils.current_stage()
        cache_dir = stage_config["cache_dir"] if use_cache else None
        msc_utils.time_stamp(stage + ".build", False)
        run_config = stage_config.get("run_config", {})
        self._logger.debug(
            "Create runner({}) by {}({})".format(stage, stage_config["runner"].__name__, run_config)
        )
        if self._runner:
            self._runner.destory()
        runner = stage_config["runner"](self._relax_mod, logger=self._logger, **run_config)
        runner.build(cache_dir=cache_dir)
        self._report["info"][stage + "_by"] = "{}({})".format(runner.framework, runner.device)
        return runner


class MSCManager(BaseManager):
    """Normal manager in MSC"""

    def update_config(self, model, config):
        config = super().update_config(model, config)
        for stage in ["prepare", "parse"]:
            if stage not in config:
                config[stage] = {}
        if config["model_type"] == MSCFramework.TORCH:
            assert isinstance(
                model, torch.nn.Module
            ), "Model for torch should be nn.Module, get {}({})".format(model, type(model))
            assert "inputs" in config, "inputs should be given to parse torch model"
            config["prepare"]["runner"] = TorchRunner.run_native
            config["parse"]["parser"] = from_torch
            parse_config = config["parse"].get("parse_config", {})
            parse_config.update(
                {
                    "input_info": [[i[1], i[2]] for i in config["inputs"]],
                    "input_names": [i[0] for i in config["inputs"]],
                }
            )
            config["parse"]["parse_config"] = parse_config
        elif config["model_type"] == MSCFramework.TENSORFLOW:
            assert isinstance(
                model, tf_v1.GraphDef
            ), "Model for tenosrflow should be tf.GraphDef, get {}({})".format(model, type(model))
            config["prepare"]["runner"] = TensorflowRunner.run_native
            config["parse"]["parser"] = from_tensorflow
            parse_config = config["parse"].get("parse_config", {})
            parse_config.update(
                {
                    "shape_dict": {i[0]: i[1] for i in config["inputs"]},
                    "outputs": config["outputs"],
                }
            )
            config["parse"]["parse_config"] = parse_config
        else:
            raise Exception("Unexpect model_type " + str(config["model_type"]))
        for stage in ["baseline", "optimize", "compile"]:
            self.update_runner(config, stage, model)
        return config

    def update_runner(self, config: dict, stage: str, model: Any) -> dict:
        """Update runner in stage config.

        Parameters
        ----------
        config: dict
            The config of a pipeline.
        stage: str
            The stage to be updated
        model: Any
            The raw model in framwork.
        """

        if stage not in config:
            return
        model_type = config["model_type"]
        if "run_type" not in config[stage]:
            config[stage]["run_type"] = model_type
        run_type = config[stage]["run_type"]
        # define runner
        if run_type == MSCFramework.TVM:
            config[stage]["runner"] = TVMRunner
        elif run_type == MSCFramework.TORCH:
            config[stage]["runner"] = TorchRunner
        elif run_type == MSCFramework.TENSORFLOW:
            config[stage]["runner"] = TensorflowRunner
        elif run_type == MSCFramework.TENSORRT:
            config[stage]["runner"] = TensorRTRunner
        else:
            raise Exception("Unexpect run_type " + str(run_type))

        # update run config
        run_config = config[stage].get("run_config", {})
        if "translate_config" not in run_config:
            run_config["translate_config"] = {}
        if "build" not in run_config["translate_config"]:
            run_config["translate_config"]["build"] = {}
        if "generate_config" not in run_config:
            run_config["generate_config"] = {}
        run_config["generate_config"].update(
            {
                "build_folder": msc_utils.get_build_dir().create_dir(stage),
            }
        )
        run_config["translate_config"]["build"]["input_aliases"] = [i[0] for i in config["inputs"]]
        run_config["translate_config"]["build"]["output_aliases"] = config["outputs"]
        if model_type == MSCFramework.TORCH:
            parameters = list(model.parameters())
            if parameters:
                ref_device = parameters[0].device
                if ref_device.type == "cpu":
                    device = "cpu"
                else:
                    device = "{}:{}".format(ref_device.type, ref_device.index)
            else:
                device = "cpu"
            run_config.update({"device": device, "is_training": model.training})
        config[stage]["run_config"] = run_config
