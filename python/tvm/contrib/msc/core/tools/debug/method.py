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
"""tvm.contrib.msc.core.tools.debug.method"""

from typing import List, Dict
import numpy as np

from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


def prune_axis(data: np.ndarray, axis: int, indices: List[int]) -> np.ndarray:
    """Delete indices on axis

    Parameters
    ----------
    data: np.ndarray
        The source data.
    axis: int
        The axis to prune
    indices: list<int>
        The indices to be pruned

    Returns
    -------
    data: np.ndarray
        The pruned data.
    """

    left_datas = [
        d for idx, d in enumerate(np.split(data, data.shape[axis], axis)) if idx in indices
    ]
    return np.concatenate(left_datas, axis=axis)


class DebugMethod(object):
    """Default debug method"""

    @classmethod
    def save_compared(
        cls,
        debugger: BaseTool,
        name: str,
        data: np.ndarray,
        compare_to: Dict[str, List[str]],
    ) -> np.ndarray:
        """compare and save the data

        Parameters
        ----------
        debugger: BaseDebugger
            The debugger
        name: str
            The name of the weight.
        data: np.ndarray
            The source data.
        stage: str
            The current stage of tool.
        compare_to: dict
            The compare config
        dataset: MSCDirectory
            The root dir

        Returns
        -------
        plan: dict
            The plan of the tensor.
        """

        config = {}
        return config

    @classmethod
    def framework(cls):
        return MSCFramework.MSC

    @classmethod
    def tool_type(cls):
        return ToolType.DEBUG


msc_utils.register_tool_method(DebugMethod)
