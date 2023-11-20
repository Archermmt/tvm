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

        config = {"info": msc_utils.inspect_array(data)}
        # save the data
        debugger._saver.save_datas({name: data}, debugger._forward_cnt)
        debugger._logger.debug("Save(%s) %s: %s", debugger._stage, name, msc_utils.MSCArray(data))
        # compare datas
        if debugger._stage in compare_to:
            diffs = {}
            for stage in compare_to[debugger._stage]:
                if stage in debugger._loaders:
                    golden = debugger._loaders[stage].load_data(name, debugger._forward_cnt)
                    report = msc_utils.compare_arrays({name: golden}, {name: data})
                    if report["passed"] == 0:
                        debugger._logger.info(
                            "Diff(%s2%s) %s: %s", stage, debugger._stage, name, report["info"][name]
                        )
                    diffs[stage] = {
                        "pass": report["passed"] == 1,
                        **msc_utils.inspect_array(np.abs(golden - data)),
                    }
            config["diffs"] = diffs
        return config

    @classmethod
    def framework(cls):
        return MSCFramework.MSC

    @classmethod
    def tool_type(cls):
        return ToolType.DEBUG


msc_utils.register_tool_method(DebugMethod)
