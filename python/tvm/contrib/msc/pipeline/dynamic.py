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
from .pipeline import BasePipeline
from .worker import MSCPipeWorker


class MSCDynamic(BasePipeline):
    """Dynamic of Pipeline, process dynamic model"""

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

        return "Dynamic " + str(msg)

    @property
    def worker_cls(self):
        return MSCPipeWorker


class TorchDynamic(MSCDynamic):
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
