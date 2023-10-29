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
"""tvm.contrib.msc.core.tools.pruner"""

from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core.tools.tool import MSCToolType, MSCTool, MSCToolImpl
from tvm.contrib.msc.core import utils as msc_utils


class BasePrunerImpl(MSCToolImpl):
    @classmethod
    def tool_type(cls):
        return MSCToolType.PRUNE


class BasePruner(MSCTool):
    def setup(self, options: dict):
        """Setup the tool

        Parameters
        ----------
        options: dict
            The options for setup the tool
        """

        super().setup(options)
        # build weight graphs
        if "prunable_types" in self._options:
            prunable_types = self._options["prunable_types"]
        else:
            prunable_types = {
                "nn.conv2d": ["weight"],
                "msc.conv2d_bias": ["weight"],
                "nn.dense": ["weight"],
            }

        if "relation_types" in self._options:
            relation_types = self._options["relation_types"]
        else:
            relation_types = {
                "reshape": "inject",
                "add": "elemwise",
                "substract": "elemwise",
                "multiply": "elemwise",
                "divide": "elemwise",
            }
        self._weight_graphs = [
            _ffi_api.WeightGraph(graph, prunable_types, relation_types) for graph in self._graphs
        ]
        # Save weight graphs for debug
        for w_graph in self._weight_graphs:
            w_graph.visualize(msc_utils.get_debug_dir().relpath(w_graph.name + ".prototxt"))

    @classmethod
    def tool_type(cls):
        return MSCToolType.PRUNE


class DefaultPruner(BasePruner):
    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultPruner)
