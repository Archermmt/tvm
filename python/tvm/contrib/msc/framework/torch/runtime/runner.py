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

import numpy as np
from typing import Dict, List, Union, Tuple

import torch
import tvm
from tvm.contrib.msc.core.runtime import ModelRunner
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.framework.torch.codegen import to_torch
from tvm.contrib.msc.framework.torch.frontend import set_weight_alias


class TorchRunner(ModelRunner):
    """Runner of Torch"""

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """
        graphs, weights = super()._translate()
        return [set_weight_alias(graphs[0])], weights

    def _to_device(self, model: torch.nn.Module, device: str) -> object:
        """Place model on device

        Parameters
        -------
        model: torch.nn.Module
            The runnable model.
        device: str
            The device for place model

        Returns
        -------
        model: torch.nn.Module
            The runnable model
        """

        if device == "cpu":
            pass
        elif device == "gpu":
            model = model.to(torch.cuda())
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        return model

    def _run_model(
        self, model: torch.nn.Module, inputs: Dict[str, np.ndarray], device: str
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Run the model to get outputs

        Parameters
        -------
        model: torch.nn.Module
            The runnable model.
        inputs: dict<str, data>
            The inputs in dict.
        device: str
            The device.

        Returns
        -------
        outputs: list<torch.Tensor>
            The outputs in list.
        """

        model_inputs = self.get_inputs()
        parameters = list(model.parameters())
        if parameters:
            in_dev = parameters[0].device
        elif device == "cpu":
            in_dev = torch.cpu()
        elif device == "gpu":
            in_dev = torch.cuda()
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        torch_inputs = [torch.from_numpy(inputs[i["name"]]).to(in_dev) for i in model_inputs]
        return model(*torch_inputs)

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
            return torch.cuda.is_available()
        return False

    @property
    def codegen_func(self):
        return to_torch

    @property
    def framework(self):
        return MSCFramework.TORCH
