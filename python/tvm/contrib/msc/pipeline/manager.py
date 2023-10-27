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

import shutil
from typing import Dict, Tuple
import numpy as np
import logging
import torch

import tvm
from tvm.relax.transform import BindParams
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tvm.runtime import TVMRunner
from tvm.contrib.msc.framework.torch.frontend import from_torch
from tvm.contrib.msc.framework.torch.runtime import TorchRunner
from tvm.contrib.msc.framework.tensorflow import tf_v1
from tvm.contrib.msc.framework.tensorflow.frontend import from_tensorflow
from tvm.contrib.msc.framework.tensorflow.runtime import TensorflowRunner
from tvm.contrib.msc.framework.tensorrt.runtime import TensorRTRunner
from .message import msg_block, time_stamp


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
        for phase in ["inputs", "outputs", "dataset", "prepare", "compile"]:
            assert phase in config, "{} should be given to run the pipeline".format(phase)
        self._model = model
        self._config = config
        workspace = msc_utils.set_workspace(config.get("workspace"), keep_history=False)
        self._config["workspace"] = workspace.path
        log_path = config.get("log_path") or workspace.relpath("MSC_LOG")
        verbose = config.get("verbose", "info")
        if verbose == "debug":
            log_level = logging.DEBUG
        elif verbose == "info":
            log_level = logging.INFO
        elif verbose == "warn":
            log_level = logging.WARN
        else:
            raise Exception("Unexcept verbose {}, should be debug| info| warn")
        self._logger = msc_utils.set_global_logger(log_level, log_path)
        time_stamp("Init", self._logger)
        self._logger.info(msg_block("MSC_CONFIG", self._config))
        self._relax_mod, self._runner = None, None

    def run_pipe(self) -> dict:
        """Run the pipeline and return object.

        Returns
        -------
        summary:
            The pipeline summary.
        """

        self.prepare()
        self._relax_mod = self.parse()
        if "optimize" in self._config:
            self.optimize()
        self.compile()
        return self.summary()

    def prepare(self):
        """Prepare datas for the pipeline."""

        time_stamp("Prepare", self._logger)
        # create loader
        loader = self._config["dataset"].get("loader")
        assert loader, "Dataset loader should be given for msc pipeline"
        if loader.startswith("from_random"):

            def get_inputs(max_num=5):
                for _ in range(max_num):
                    yield {i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]}

            loader = get_inputs
        elif msc_utils.is_dataset(self._config["dataset"]):

            def get_inputs(max_num=-1):
                for _ in range(max_num):
                    yield {i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]}

            loader = get_inputs
        assert callable(loader), "Loader {} is not callable".format(loader)

        # save golden
        golden_folder, golden_cnt = msc_utils.get_dataset_dir().relpath("Golden"), 0
        if "runner" in self._config["prepare"]:
            max_num = self._config["dataset"].get("max_num", 5)
            input_names = [i[0] for i in self._config["inputs"]]
            with msc_utils.MSCDataSaver(
                golden_folder, input_names, self._config["outputs"]
            ) as saver:
                for inputs in loader():
                    if golden_cnt >= max_num:
                        break
                    outputs = self._config["prepare"]["runner"](
                        self._model, inputs, input_names, self._config["outputs"]
                    )
                    golden_cnt = saver.save(inputs, outputs)
            self._logger.info("Saved {} datas as golden -> {}".format(golden_cnt, golden_folder))
        elif "golden" in self._config["prepare"]:
            src_golden = self._config["prepare"]["golden"]
            assert msc_utils.is_dataset(src_golden), "Golden folder {} is not msc dataset".format(
                src_golden
            )
            shutil.copytree(src_golden, golden_folder)
            self._logger.info("Copy golden {} ->{}".format(src_golden, golden_folder))
        else:
            raise Exception("golden or runner should given in prepare to save golden")

    def parse(self) -> Tuple[tvm.IRModule, Dict[str, tvm.nd.array]]:
        """Parse the model to IRModule.

        Returns
        -------
        mod: tvm.IRModule
            The translated IRModule.
        params: dict of <string:tvm.ndarray>
            The params from the IRModule.
        """

        time_stamp("Parse", self._logger)
        parse_config = self._config["parse"].get("config", {})
        mod, params = self._config["parse"]["parser"](self._model, as_msc=False, **parse_config)
        if params:
            mod = BindParams("main", params)(mod)
        return mod

    def optimize(self, ret_type: str = "model") -> object:
        """Run the optimize and return object.

        Parameters
        ----------
        ret_type: str
            The return type runner| model.

        Returns
        -------
        holder:
            The runner or model.
        """

        time_stamp("Optimize", self._logger)
        run_config = self._config["optimize"].get("config", {})
        self._runner = self._config["optimize"]["runner"](
            self._relax_mod, logger=self._logger, **run_config
        )
        self._runner.build()
        return self.get_return(ret_type)

    def compile(self, ret_type: str = "model") -> object:
        """Run the compile and return object.

        Parameters
        ----------
        ret_type: str
            The return type runner| model.

        Returns
        -------
        holder:
            The runner or model.
        """

        time_stamp("Compile", self._logger)
        run_config = self._config["compile"].get("config", {})
        self._runner = self._config["compile"]["runner"](
            self._relax_mod, logger=self._logger, **run_config
        )
        self._runner.build()
        return self.get_return(ret_type)

    def get_return(self, ret_type: str = "runner") -> object:
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


class MSCManager(BaseManager):
    """Normal manager in MSC"""

    def __init__(self, model, config):
        assert "model_type" in config, "model_type should be given for msc pipeline"
        for phase in ["parse", "prepare"]:
            if phase not in config:
                config[phase] = {}
        if config["model_type"] == MSCFramework.TORCH:
            assert isinstance(
                model, torch.nn.Module
            ), "Model for torch should be nn.Module, get {}({})".format(model, type(model))
            config["prepare"]["runner"] = TorchRunner.run_native
            config["parse"]["parser"] = from_torch
            parse_config = config["parse"].get("config", {})
            assert "inputs" in config, "inputs should be given to parse torch model"
            parse_config.update(
                {
                    "input_info": [[i[1], i[2]] for i in config["inputs"]],
                    "input_names": [i[0] for i in config["inputs"]],
                }
            )
            config["parse"]["config"] = parse_config
            for phase in ["optimize", "compile"]:
                if phase in config:
                    run_config = config[phase].get("config", {})
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
                    config[phase]["config"] = run_config
        elif config["model_type"] == MSCFramework.TENSORFLOW:
            assert isinstance(
                model, tf_v1.GraphDef
            ), "Model for tenosrflow should be tf.GraphDef, get {}({})".format(model, type(model))
            config["prepare"]["runner"] = TensorflowRunner.run_native
            config["parse"]["parser"] = from_tensorflow
            parse_config = config["parse"].get("config", {})
            assert "inputs" in config, "inputs should be given to parse torch model"
            assert "outputs" in config, "outputs should be given to parse torch model"
            parse_config.update(
                {
                    "shape_dict": {i[0] + ":0": i[1] for i in config["inputs"]},
                    "outputs": config["outputs"],
                }
            )
            config["parse"]["config"] = parse_config
        else:
            raise Exception("Unexpect model_type " + str(config["model_type"]))
        for phase in ["optimize", "compile"]:
            if "type" in config.get(phase, {}):
                run_type = config[phase].pop("type")
                if run_type == MSCFramework.TVM:
                    config[phase]["runner"] = TVMRunner
                elif run_type == MSCFramework.TORCH:
                    config[phase]["runner"] = TorchRunner
                elif run_type == MSCFramework.TENSORFLOW:
                    config[phase]["runner"] = TensorflowRunner
                elif run_type == MSCFramework.TENSORRT:
                    config[phase]["runner"] = TensorRTRunner
                else:
                    raise Exception("Unexpect run_type " + str(run_type))
        super().__init__(model, config)
