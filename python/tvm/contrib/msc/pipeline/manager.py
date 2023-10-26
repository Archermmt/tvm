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

from typing import Dict, Tuple
import numpy as np
import logging

import tvm
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tvm.runtime import TVMRunner
from tvm.contrib.msc.framework.torch.frontend import from_torch
from tvm.contrib.msc.framework.torch.runtime import TorchRunner
from tvm.contrib.msc.framework.tensorflow.frontend import from_tensorflow
from tvm.contrib.msc.framework.tensorflow.runtime import TensorflowRunner
from tvm.contrib.msc.framework.tensorrt.runtime import TensorRTRunner
from .message import log_block, time_stamp


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
        log_block("MSC_CONFIG", self._config, self._logger)
        self._runner = None

    def run_pipe(self, ret_type: str = "runner") -> object:
        """Run the pipeline and return object.

        Parameters
        ----------
        ret_type: str
            The return type runner| model.

        Returns
        -------
        holder:
            The runner or model.
        """

        self.prepare()
        self._mod, self._params = self.parse()

        if "optimize" in self._config:
            self.optimize()

        return self.compile()

    def prepare(self):
        """Prepare datas for the pipeline."""

        time_stamp("Check Config", self._logger)
        assert "inputs" in self._config, "inputs should be given to run the pipeline"
        assert "outputs" in self._config, "outputs should be given to run the pipeline"
        assert "dataset" in self._config, "dataset should be config to run the manager"
        assert "compile" in self._config, "compile should be config to run the manager"
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
        if "type" in config.get("parse", {}):
            parse_type = config["parse"].pop("type")
            if parse_type == MSCFramework.TORCH:
                config["parse"]["parser"] = from_torch
            elif parse_type == MSCFramework.TENSORFLOW:
                config["parse"]["parser"] = from_tensorflow
            else:
                raise Exception("Unexpect parse_type " + str(parse_type))
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
