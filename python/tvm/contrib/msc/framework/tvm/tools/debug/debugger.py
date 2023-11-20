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
"""tvm.contrib.msc.framework.tvm.tools.debug.debugger"""

from typing import List, Union

import tvm
from tvm.contrib.msc.core.tools.tool import ToolType, Strategy
from tvm.contrib.msc.core.tools.debug import BaseDebugger
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TVMDebuggerFactory(object):
    """Debugger factory for tvm"""

    def create(self, base_cls: BaseDebugger):
        class debugger(base_cls):
            """Adaptive debugger for tvm"""

            def _execute_before_build(self, block_builder: tvm.relax.BlockBuilder):
                """Execute before model build

                Parameters
                ----------
                block_builder: tvm.relax.BlockBuilder
                    The block builder.
                """

                self._block_builder = block_builder
                self._output_vars, self._debug_names = {}, []

            def _execute_after_build(
                self, output: Union[tvm.relax.Var, List[tvm.relax.DataflowVar]]
            ) -> List[tvm.relax.Var]:
                """Execute after model build

                Parameters
                ----------
                output: var or list<var>
                    The output var of the model.

                Returns
                -------
                outputs: list<var>
                    The modified outputs var.
                """

                self._debug_names = list(sorted(self._output_vars.keys()))
                debug_vars = [self._output_vars[o]["var"] for o in self._debug_names]
                if isinstance(output, tvm.relax.Var):
                    return [output] + debug_vars
                return output + debug_vars

            def _execute_after_forward(
                self, outputs: List[tvm.runtime.NDArray]
            ) -> Union[tvm.runtime.NDArray, List[tvm.runtime.NDArray]]:
                """Execute after model forward

                Parameters
                ----------
                outputs: list<np.ndarray>
                    The output datas.

                Returns
                -------
                output: np.ndarray or list<np.ndarray>
                    The modified output ndarray.
                """

                output_num = len(outputs) - len(self._debug_names)
                for data, name in zip(outputs[output_num:], self._debug_names):
                    self._debug_tensor(data, name, self._output_vars[name]["strategy"])
                if output_num == 1:
                    return outputs[0]
                return outputs[:output_num]

            def _process_tensor(
                self,
                tensor: tvm.relax.DataflowVar,
                name: str,
                consumer: str,
                scope: str,
                strategy: Strategy,
            ) -> tvm.relax.DataflowVar:
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

                if self.is_weight(name):
                    return self._debug_tensor(self.get_data(name), name, strategy)
                if name not in self._output_vars:
                    self._output_vars[name] = {"strategy": strategy, "var": tensor}
                    self._debug_names.append(name)
                return tensor

            @classmethod
            def framework(cls):
                return MSCFramework.TVM

        return debugger


factory = TVMDebuggerFactory()
tools = msc_utils.get_registered_tool_cls(MSCFramework.MSC, ToolType.DEBUG, tool_style="all")
for tool in tools.values():
    msc_utils.register_tool_cls(factory.create(tool))
