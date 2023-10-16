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
"""tvm.contrib.msc.core.runtime.runner"""

from typing import Dict, Optional, Any, List, Tuple

import tvm
from tvm.contrib.msc.core.ir import MSCGraph, from_relax
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class BaseRunner(object):
    """Basic runner of MSC

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    tools_config: dict
        The config of MSC Tools.
    translate_config: dict
        The config for translate IRModule to MSCGraph.
    codegen_config: dict
        The config for build MSCGraph to runnable model.
    name: str
        The name of the runner
    device: str
        The device of the model, cpu| gpu
    """

    def __init__(
        self,
        mod: tvm.IRModule,
        tools_config: Optional[Dict[str, Any]] = None,
        translate_config: Optional[Dict[str, str]] = None,
        load_config: Optional[Dict[str, str]] = None,
        name: str = "main",
        device: str = "cpu",
    ):
        self._mod = mod
        self._tools_config = tools_config
        self._translate_config = translate_config
        self._load_config = load_config
        self._name = name
        self._device = device if self._device_enabled(device) else "cpu"
        self._graphs, self._weights = [], []
        self._model = None
        self._logger = msc_utils.get_global_logger()

    def build(self, build_graph: bool = False) -> object:
        """Build the runnable object

        Parameters
        -------
        build_graph: bool
            Whether to build the MSCGraphs.

        Returns
        -------
        model: object
           The runnable object.
        """

        # Get or rebuild graphs
        if build_graph or not self._graphs:
            self._graphs, self._weights = self._translate()
            self._logger.info("Translate {} graphs from module".format(len(self._graphs)))

        # create tools
        if self._tools_config:
            raise NotImplementedError("Build runner with tools is not supported")

        # load model
        model = self._load()
        if "post_loader" in self._load_config:
            loader = self._load_config["post_loader"]
            load_config = self._load_config.get("post_load_config")
            model = loader(**load_config)
            self._logger.info(
                "Model({}) processed by post_loader {}({})".format(
                    self.framework, loader, load_config
                )
            )
        self._model = self._to_device(model, self._device)
        self._logger.info("Model({}) loaded on device {}".format(self.framework, len(self._device)))
        return self._model

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        raise NotImplementedError("_translate is not implemented for " + str(self.__class__))

    def _load(self) -> object:
        """Codegen the model according to framework

        Returns
        -------
        model: object
            The runnable model
        """

        raise NotImplementedError("_load is not implemented for " + str(self.__class__))

    def _to_device(self, model: object, device: str) -> object:
        """Place model on device

        Parameters
        -------
        model: object
            The runnable model on cpu.
        device: str
            The device for place model

        Returns
        -------
        model: object
            The runnable model
        """

        raise NotImplementedError("_to_device is not implemented for " + str(self.__class__))

    def _device_enabled(self, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        return True

    @property
    def codegen_func(self):
        raise NotImplementedError("codegen_func is not implemented for " + str(self.__class__))

    @property
    def framework(self):
        return MSCFramework.MSC


class ModelRunner(BaseRunner):
    """Model runner of MSC"""

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        graph, weights = from_relax(
            self._mod,
            trans_config=self._translate_config.get("transform"),
            build_config=self._translate_config.get("build"),
            opt_config=self._translate_config.get("optimize"),
        )
        return [graph], [weights]

    def _load(self):
        """Codegen the model according to framework

        Returns
        -------
        model: object
            The runnable model
        """

        return self.codegen_func(
            self._graphs[0],
            self._weights[0],
            codegen_config=self._load_config.get("codegen"),
            print_config=self._load_config.get("build"),
            build_folder=msc_utils.get_build_dir(),
        )


class BYOCRunner(BaseRunner):
    """BYOC runner of MSC"""

    def __init__(
        self,
        mod: tvm.IRModule,
        tools_config: Optional[Dict[str, Any]] = None,
        translate_config: Optional[Dict[str, str]] = None,
        load_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(mod, tools_config, translate_config, load_config)
        self._byoc_mod, self._graph_infos = None, {}

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        self._byoc_mod, self._graph_infos = self.partition_func(
            "msc_" + str(self.framework),
            self._mod,
            trans_config=self._translate_config.get("transform"),
            build_config=self._translate_config.get("build"),
            opt_config=self._translate_config.get("optimize"),
        )
        graphs, weights = [], []
        for _, graph, sub_weights in self._graph_infos:
            graphs.append(graph)
            weights.append(sub_weights)
        return graphs, weights

    def _load(self):
        """Codegen the model according to framework

        Returns
        -------
        model: object
            The runnable model
        """

        mod = self.codegen_func(
            self._graph_infos,
            codegen_config=self._load_config.get("codegen"),
            print_config=self._load_config.get("build"),
            build_folder=msc_utils.get_build_dir(),
            output_folder=msc_utils.get_output_dir(),
        )
        return mod

    def _to_device(self, model: object, device: str) -> object:
        """Place model on device

        Parameters
        -------
        model: object
            The runnable model on cpu.
        device: str
            The device for place model

        Returns
        -------
        model: object
            The runnable model
        """

        model = tvm.relax.transform.LegalizeOps()(model)
        if device == "cpu":
            target = tvm.target.Target("llvm")
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cpu())
        elif device == "gpu":
            target = tvm.target.Target("cuda")
            with target:
                model = tvm.tir.transform.DefaultGPUSchedule()(model)
            with tvm.transform.PassContext(opt_level=3):
                exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(exec, tvm.cuda())
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        return runnable

    def _device_enabled(self, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        if device == "cpu":
            return True
        if device == "gpu":
            return tvm.cuda().exist
        return False

    @property
    def partition_func(self):
        raise NotImplementedError("partition_func is not implemented for " + str(self.__class__))
