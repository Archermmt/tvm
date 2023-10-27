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
"""tvm.contrib.msc.framework.torch.runtime.runner"""

import numpy as np
from typing import Dict, List, Union, Tuple

import torch
import tvm
from tvm.contrib.msc.core.runtime import ModelRunner
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.framework.torch.codegen import to_torch
from tvm.contrib.msc.framework.torch.frontend import set_weight_alias
from tvm.contrib.msc.core import utils as msc_utils


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

    def _to_device(self, model: object, device: str, is_training: bool) -> object:
        """Place model on device

        Parameters
        -------
        model: object
            The runnable model on cpu.
        device: str
            The device for place model
        is_training: bool
            Whether to load model for training

        Returns
        -------
        model: object
            The runnable model
        """

        if device == "cpu":
            pass
        elif device == "gpu":
            model = model.to(torch.device("cuda:0"))
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        if is_training:
            model = model.train()
        else:
            model = model.eval()
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

    @classmethod
    def run_native(
        cls,
        model: torch.nn.Module,
        inputs: Dict[str, np.ndarray],
        input_names: List[str],
        output_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """Run the datas and get outputs

        Parameters
        -------
        model: torch.nn.Module
            The runnable model.
        inputs: dict<str, data>
            The inputs in dict.
        input_names: list<str>
            The input names.
        output_names: list<str>
            The outut names.

        Returns
        -------
        outputs: dict<str, np.array>
            The outputs in dict.
        """

        parameters = list(model.parameters())
        if parameters:
            device = parameters[0].device
        else:
            device = torch.device("cpu")
        torch_inputs = [torch.from_numpy(inputs[i_name]).to(device) for i_name in input_names]
        outputs = model(*torch_inputs)
        if isinstance(outputs, torch.Tensor):
            assert len(output_names) == 1, "Expect 1 outputs, get " + str(output_names)
            return {output_names[0]: msc_utils.cast_array(outputs)}
        assert len(output_names) == len(outputs), "Outputs mismatch, {} with {}".format(
            output_names, len(outputs)
        )
        return {
            o_name: msc_utils.cast_array(o_data) for o_name, o_data in zip(output_names, outputs)
        }
