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
"""tvm.contrib.msc.core.tools.quantize.quantizer"""

from typing import List, Tuple, Dict, Any

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool, Strategy
from tvm.contrib.msc.core import utils as msc_utils


class BaseQuantizer(BaseTool):
    """Base quantizer for all"""

    def reset(
        self,
        graphs: List[MSCGraph],
        weights: List[Dict[str, tvm.nd.array]],
        cache_dir: msc_utils.MSCDirectory = None,
    ) -> Tuple[List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
        """Reset the tool with graphs and weights

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        cache_dir: MSCDirectory
            cache path for save/load info

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        """

        if self._plan:
            self._calibrated = True
        else:
            self._calibrated = False
            self._calibrate_plan = {}
            self.change_stage("gather")
        return super().reset(graphs, weights, cache_dir)

    def calibrate(self) -> dict:
        """Calibrate the datas

        Returns
        -------
        plan: dict
            The calibrated plan.
        """

        new_plan = {}
        self.change_stage("calibrate")
        for tensor_id, plan in self._calibrate_plan.items():
            if plan.get("calibrated", False):
                new_plan[tensor_id] = plan
                continue
            name, consumer = self.from_tensor_id(tensor_id)
            strategy = self._get_tensor_strategy(name, consumer)
            new_plan[tensor_id] = strategy(self, name, consumer, plan)
        if any(not plan.get("calibrated", False) for plan in new_plan.values()):
            self._calibrate_plan = new_plan
            self.change_stage("gather")
        else:
            self._calibrated = True
            for name, plan in new_plan.items():
                self._plan[name] = {k: v for k, v in plan.items() if k not in ("calibrated")}
            self.change_stage("quantize")
        return new_plan

    def _check_tensor(self, name: str, consumer: str, strategy: Strategy) -> bool:
        """Check if the tensor should be processed

        Parameters
        -------
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        strategy: Strategy
            The strategy for the tensor

        Returns
        -------
        vaild: bool
            Whether to process the tensor.
        """

        # only process tensor with nbits>0
        if strategy.get_config("nbits", 8) == -1:
            return False
        return True

    def _process_tensor(self, tensor: Any, name: str, consumer: str, strategy: Strategy) -> Any:
        """Process tensor

        Parameters
        -------
        tensor: Any
            Tensor in framework
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        strategy: Strategy
            The strategy for the tensor

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        if not self.calibrated:
            return self._gather_tensor(tensor, name, consumer, strategy)
        return self._quantize_tensor(tensor, name, consumer, strategy)

    def _gather_tensor(self, tensor: Any, name: str, consumer: str, strategy: Strategy) -> Any:
        """Calibrate tensor

        Parameters
        -------
        tensor: Any
            Tensor in framework
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        strategy: Strategy
            The strategy for the tensor

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        tensor_id = self.to_tensor_id(name, consumer)
        plan = self._calibrate_plan.get(tensor_id, {})
        if plan.get("calibrated", False):
            return tensor
        self._calibrate_plan[tensor_id] = strategy(self, tensor, name, consumer, plan)
        return tensor

    def _quantize_tensor(self, tensor: Any, name: str, consumer: str, strategy: Strategy) -> Any:
        """Quantize tensor

        Parameters
        -------
        tensor: Any
            Tensor in framework
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        strategy: Strategy
            The strategy for the tensor

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        tensor_id = self.to_tensor_id(name, consumer)
        return strategy(self, tensor, name, consumer, **self.get_plan(tensor_id))

    @property
    def calibrated(self):
        return self._calibrated

    @classmethod
    def tool_type(cls):
        return ToolType.QUANTIZE


class DefaultQuantizer(BaseQuantizer):
    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultQuantizer)
