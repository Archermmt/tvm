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
"""tvm.contrib.msc.core.tools.debug.debugger"""

from typing import Any
from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool, Strategy
from tvm.contrib.msc.core import utils as msc_utils


class BaseDebugger(BaseTool):
    """Base pruner for all"""

    def setup(self, options: dict):
        """Setup the tool

        Parameters
        ----------
        options: dict
            The options for setup the tool
        """

        options = options or {}
        super().setup(options)
        data_folder = msc_utils.get_dataset_dir().create_dir("Debug")
        self._loaders = {}
        for folder in data_folder.listdir():
            if msc_utils.is_simple_dataset(data_folder.relpath(folder)):
                self._loaders[folder] = msc_utils.SimpleDataLoader(data_folder.relpath(folder))
        self._saver = msc_utils.SimpleDataSaver(data_folder.relpath(self._stage))
        self._max_iter = options.get("max_iter", 1)

    def destory(self):
        """Destory tool"""

        self._saver.finalize()
        super().destory()

    def _execute_after_forward(self, output: Any) -> Any:
        """Execute after model forward

        Parameters
        ----------
        output: Any
            The output reference of the model.

        Returns
        -------
        output: Any
           The modified output reference.
        """

        if self._forward_cnt < self._max_iter:
            passed_info = {}
            for info in self._plan.values():
                if "diffs" not in info[self._stage]:
                    continue
                for stage, p_info in info[self._stage]["diffs"].items():
                    if stage not in passed_info:
                        passed_info[stage] = {"total": 0, "passed": 0}
                    passed_info[stage]["total"] += 1
                    if p_info["pass"]:
                        passed_info[stage]["passed"] += 1
            msg = "Debug({}) {} datas".format(self._stage, len(self._plan))
            for stage, p_info in passed_info.items():
                msg += " {} passed {}/{}".format(stage, p_info["passed"], p_info["total"])
            self._logger.info(msg)
        return output

    def _debug_tensor(self, tensor: Any, name: str, strategy: Strategy) -> Any:
        """Process tensor

        Parameters
        -------
        tensor: Any
            Tensor in framework
        name: str
            The name of the tensor.
        strategy: Strategy
            The strategy for the tensor

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        if self._stage in self._plan.get(name, {}):
            return tensor
        if name not in self._plan:
            self._plan[name] = {}
        self._plan[name][self._stage] = strategy(self, name, msc_utils.cast_array(tensor))
        return tensor

    @classmethod
    def tool_type(cls):
        return ToolType.DEBUG


class DefaultDebugger(BaseDebugger):
    def _check_tensor(self, name: str, consumer: str, scope: str, strategy: Strategy) -> bool:
        """Check if the tensor should be processed

        Parameters
        -------
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        scope: str
            The scope mark teacher| student| null
        strategy: Strategy
            The strategy for the tensor

        Returns
        -------
        vaild: bool
            Whether to process the tensor.
        """

        if self._forward_cnt >= self._max_iter:
            return False
        compare_to = strategy.get_config("compare_to", {})
        if self._stage in compare_to:
            return True
        for stages in compare_to.values():
            if self._stage in stages:
                return True
        return False

    def _process_tensor(
        self, tensor: Any, name: str, consumer: str, scope: str, strategy: Strategy
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
            The scope mark teacher| student| null
        strategy: Strategy
            The strategy for the tensor

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        return self._debug_tensor(tensor, name, strategy)

    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultDebugger)
