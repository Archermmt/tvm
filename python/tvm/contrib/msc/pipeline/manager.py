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

import time
from typing import Dict, Tuple
import numpy as np
import traceback
import torch

import tvm
from tvm.contrib.msc.core.utils.namespace import MSCFramework
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
    model: object
        The raw model in framwork.
    config: dict
        The config for pipeline.
    """

    def __init__(self, model, config):
        # check config
        for stage in ["inputs", "outputs", "dataset", "prepare", "compile"]:
            assert stage in config, "{} should be given to run the pipeline".format(stage)
        workspace = msc_utils.set_workspace(config.get("workspace"), keep_history=False)
        log_path = config.get("log_path") or workspace.relpath("MSC_LOG", keep_history=False)
        self._logger = msc_utils.set_global_logger(config.get("verbose", "info"), log_path)
        msc_utils.time_stamp("init", True)
        self._model = model
        config["workspace"] = workspace.path
        self._logger.info(msc_utils.msg_block("CONFIG", config))
        self._config = self.update_config(model, config)
        self._relax_mod, self._runner = None, None
        self._sample_inputs = None
        self._report = {
            "success": False,
            "workspace": workspace.path,
            "log": log_path,
            "model_type": self._config["model_type"],
            "optimize_by": self._config.get("optimize", {}).get("run_type"),
            "compile_by": self._config["compile"]["run_type"],
            "duration": {},
            "profile": {},
        }

    def update_config(self, model: object, config: dict) -> dict:
        """Update config

        Parameters
        ----------
        model: object
            The raw model in framwork.
        config: dict
            The config for pipeline.

        Returns
        -------
        config: dict
            The updated config.
        """

        return config

    def run_pipe(self) -> dict:
        """Run the pipeline and return object.

        Returns
        -------
        summary:
            The pipeline summary.
        """

        err_msg = None
        try:
            msc_utils.time_stamp("prepare", True)
            self.prepare(self._config["prepare"])
            msc_utils.time_stamp("parse", True)
            self.parse(self._config["parse"])
            if "baseline" in self._config:
                msc_utils.time_stamp("baseline", True)
                self.baseline(self._config["baseline"])
            if "optimize" in self._config:
                msc_utils.time_stamp("optimize", True)
                self.optimize(self._config["optimize"])
            msc_utils.time_stamp("compile", True)
            self.compile(self._config["compile"])
            msc_utils.time_stamp("end", True, False)
        except Exception as e:
            err_msg = "Pipeline failed:{}\nTrace: {}".format(e, traceback.format_exc())
        report = self.summary(err_msg)
        self._logger.info(msc_utils.msg_block("SUMMARY", report))
        return report

    def prepare(self, stage_config: dict):
        """Prepare datas for the pipeline.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        """

        # create loader
        loader = self._config["dataset"].get("loader")
        assert loader, "Dataset loader should be given for msc pipeline"
        if loader.startswith("from_random"):

            def get_inputs(max_num=5):
                for _ in range(max_num):
                    yield {i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]}

            data_loader = get_inputs
        elif msc_utils.is_dataset(loader):

            def get_inputs(max_num=-1):
                for _ in range(max_num):
                    yield {i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]}

            data_loader = get_inputs
        else:
            data_loader = loader
        assert callable(data_loader), "Loader {} is not callable".format(data_loader)

        # save golden
        golden_folder, golden_cnt = msc_utils.get_dataset_dir().relpath("Golden", False), 0
        max_num = self._config["dataset"].get("max_num", 5)
        input_names = [i[0] for i in self._config["inputs"]]
        report = {"golden_folder": golden_folder}
        if "runner" in stage_config:
            with msc_utils.MSCDataSaver(
                golden_folder, input_names, self._config["outputs"]
            ) as saver:
                for inputs in data_loader():
                    if golden_cnt >= max_num:
                        break
                    if not self._sample_inputs:
                        self._sample_inputs = inputs
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
                    if not self._sample_inputs:
                        self._sample_inputs = inputs
                    golden_cnt = saver.save(inputs, outputs)
                report["datas_info"] = saver.info
        else:
            raise Exception("golden or runner should given in prepare to save golden")
        report["sample_inputs"] = self._sample_inputs
        self._logger.info(msc_utils.msg_block("GOLDEN", report))
        self._logger.info("Saved {} datas as golden -> {}".format(golden_cnt, golden_folder))

        # profile
        if "profile" in stage_config and "runner" in stage_config:
            benchmark = stage_config["profile"].get("benchmark", {})
            repeat = benchmark.get("repeat", 100)
            self._logger.debug(
                "Prepare profile with {}({})".format(stage_config["runner"], benchmark)
            )
            _, avg_time = stage_config["runner"](
                self._model, inputs, input_names, self._config["outputs"], **benchmark
            )
            self._logger.info("Profile(prepare) {} times -> {:.2f} ms".format(repeat, avg_time))
            self._report["profile"]["prepare"] = {"latency": "{:.2f} ms".format(avg_time)}

    def parse(self, stage_config: dict) -> Tuple[tvm.IRModule, Dict[str, tvm.nd.array]]:
        """Parse the model to IRModule.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        """

        parse_config = stage_config.get("parse_config", {})
        self._logger.debug("Parse with {}({})".format(stage_config["parser"], parse_config))
        self._relax_mod, _ = stage_config["parser"](self._model, as_msc=False, **parse_config)

    def baseline(self, stage_config: dict):
        """Run the baseline.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        """

        self._create_runner(stage_config)
        if "profile" in stage_config:
            self._profile(stage_config)

    def optimize(self, stage_config: dict, ret_type: str = "model") -> object:
        """Run the optimize and return object.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        ret_type: str
            The return type runner| model.

        Returns
        -------
        holder:
            The runner or model.
        """

        self._create_runner(stage_config)
        if "profile" in stage_config:
            self._profile(stage_config)
        return self._get_holder(ret_type)

    def compile(self, stage_config: dict, ret_type: str = "model") -> object:
        """Run the compile and return object.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        ret_type: str
            The return type runner| model.

        Returns
        -------
        holder:
            The runner or model.
        """

        self._create_runner(stage_config)
        if "profile" in stage_config:
            self._profile(stage_config)
        return self._get_holder(ret_type)

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
        msc_utils.get_workspace().destory()

    def _profile(self, stage_config: str):
        """Profile the runner.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        """

        if "profile" not in stage_config:
            return
        stage = msc_utils.current_stage()
        msc_utils.time_stamp(stage + ".profile", False)
        profile_config = stage_config["profile"]
        if stage not in self._report["profile"]:
            self._report["profile"][stage] = {}

        # check result
        check_config = profile_config.get("check", {})
        if check_config:
            loader = msc_utils.MSCDataLoader(msc_utils.get_dataset_dir().relpath("Golden"))
            total, passed = 0, 0
            report = {}
            for idx, (inputs, outputs) in enumerate(loader):
                results = self._runner.run(inputs)
                iter_report = msc_utils.compare_arrays(outputs, results)
                total += iter_report["total"]
                passed += iter_report["passed"]
                report["iter_" + str(idx)] = iter_report["info"]
            title = "Check({}) pass {}/{}".format(stage, passed, total)
            self._logger.info(msc_utils.msg_block(title, report))
            self._report["profile"][stage]["accuracy"] = "{}/{}({:.2f}%)".format(
                passed, total, float(passed) * 100 / total
            )

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
            self._logger.info("Profile({}) {} times -> {:.2f} ms".format(stage, repeat, avg_time))
            self._report["profile"][stage]["latency"] = "{:.2f} ms".format(avg_time)

    def _get_holder(self, ret_type: str = "runner") -> object:
        """Return object by type.

        Parameters
        ----------
        ret_type: str
            The return type runner| model.

        Returns
        -------
        holder:
            The runner or model.
        """

        if ret_type == "runner":
            return self._runner
        elif ret_type == "model":
            return self._runner.get_model()
        raise Exception("Unexpect return type " + str(ret_type))

    def _create_runner(self, stage_config: dict):
        """Create runner.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        """

        stage = msc_utils.current_stage()
        msc_utils.time_stamp(stage + ".build", False)
        run_config = stage_config.get("run_config", {})
        self._logger.debug(
            "Create runner({}) with {}({})".format(stage, stage_config["runner"], run_config)
        )
        if self._runner:
            self._runner.destory()
        self._runner = stage_config["runner"](self._relax_mod, logger=self._logger, **run_config)
        self._runner.build()


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

    def update_runner(self, config: dict, stage: str, model: object) -> dict:
        """Update runner in stage config.

        Parameters
        ----------
        config: dict
            The config of a pipeline.
        stage: str
            The stage to be updated
        model: object
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
        run_config["translate_config"]["build"]["input_aliases"] = [i[0] for i in config["inputs"]]
        run_config["translate_config"]["build"]["output_aliases"] = config["outputs"]
        if model_type == MSCFramework.TORCH:
            parameters = list(model.parameters())
            if parameters:
                dev_type = parameters[0].device.type
                if dev_type == "cpu":
                    device = "cpu"
                else:
                    device = "{}:{}".format(dev_type, parameters[0].device.index)
            else:
                device = "cpu"
            run_config.update({"device": device, "is_training": model.training})
        config[stage]["run_config"] = run_config
