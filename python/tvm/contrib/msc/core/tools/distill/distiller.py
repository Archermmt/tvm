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
"""tvm.contrib.msc.core.tools.distill.distiller"""

from typing import List, Any

from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool, Strategy
from tvm.contrib.msc.core import utils as msc_utils


class BaseDistiller(BaseTool):
    """Base distiller for all"""

    def _support_scope(self, scope: str) -> bool:
        """Check if the scope si supported

        Parameters
        -------
        scope: str
            The scope mark, should be null or ToolScope

        Returns
        -------
        vaild: bool
            Whether to process the tensor.
        """

        return True

    def _process_tensor(
        self, tensor: Any, name: str, consumer: str, scope: str, strategys: List[Strategy]
    ) -> Any:
        """Process tensor

        Parameters
        -------
        tensor: Any
            Tensor in framework
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        scope: str
            The scope mark teacher| student| null.
        strategys: list<Strategy>
            The strategys for the tensor.

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        return self._distill_tensor(tensor, name, consumer, scope, strategys)

    def _distill_tensor(
        self, tensor: Any, name: str, consumer: str, scope: str, strategys: List[Strategy]
    ) -> Any:
        """Process tensor

        Parameters
        -------
        tensor: Any
            Tensor in framework
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        scope: str
            The scope mark teacher| student| null.
        strategys: list<Strategy>
            The strategys for the tensor.

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        if name not in self._plan:
            self._plan[name] = {}
        plan = {}
        for strategy in strategys:
            plan.update(strategy(self, tensor, name, consumer, scope))
        self._plan[name][scope] = plan
        return tensor

    @classmethod
    def tool_type(cls):
        return ToolType.DISTILLER


class DefaultDistiller(BaseDistiller):
    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultDistiller)
