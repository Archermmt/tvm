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

from typing import List, Dict, Iterable, Tuple, Any

import tvm
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

        super().setup(options)
        data_folder = msc_utils.get_dataset_dir().create_dir("Debug")
        self._loaders = {}
        for folder in data_folder.listdir():
            print("has sub_folder " + str(folder))
            if msc_utils.is_simple_dataset(data_folder.relpath(folder)):
                print("should add to loader")
                self._loaders[folder] = msc_utils.SimpleDataLoader(data_folder.relpath(folder))
        self._saver = msc_utils.SimpleDataSaver(data_folder.relpath(self._stage))

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

        print("chekcing {} with {}".format(name, strategy))
        raise Exception("stop here!!")
        return True

    @classmethod
    def tool_type(cls):
        return ToolType.DEBUG


class DefaultDebugger(BaseDebugger):
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

        return tensor

    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultDebugger)
