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
"""tvm.contrib.msc.core.tools.prune.pruner"""

from typing import List, Dict, Iterable, Tuple

import tvm
from tvm.contrib.msc.core.ir import MSCGraph, WeightJoint, MSCTensor
from tvm.contrib.msc.core.tools.tool import MSCToolType, MSCTool, MSCToolImpl
from tvm.contrib.msc.core.utils.namespace import MSCMap, MSCKey
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils
from .method import prune_axis


class MSCPrunerImpl(MSCToolImpl):
    @classmethod
    def tool_type(cls):
        return MSCToolType.PRUNE


class MSCPruner(MSCTool):
    """Base pruner for all"""

    def reset(self, graphs: List[MSCGraph], weights: List[Dict[str, tvm.nd.array]]):
        """Reset the tool

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        """

        # Build weight graphs
        if "prunable_types" in self._options:
            self._prunable_types = self._options["prunable_types"]
        else:
            self._prunable_types = {
                "nn.conv2d": ["weight"],
                "msc.conv2d_bias": ["weight"],
                "msc.linear": ["weight"],
                "msc.linear_bias": ["weight"],
            }

        if "relation_types" in self._options:
            relation_types = self._options["relation_types"]
        else:
            relation_types = {
                "concatenate": "multi_inputs",
                "reshape": "reshape",
                "add": "passby",
                "substract": "passby",
                "multiply": "passby",
                "divide": "passby",
            }
        self._weight_graphs = [
            _ffi_api.WeightGraph(graph, self._prunable_types, relation_types) for graph in graphs
        ]
        # Save weight graphs for debug
        if MSCMap.get(MSCKey.ON_DEBUG, False):
            for w_graph in self._weight_graphs:
                w_graph.visualize(msc_utils.get_debug_dir().relpath(w_graph.name + ".prototxt"))

        if not self._runtime_config:
            return super().reset(graphs, weights)

        # Prune the weights
        graphs, weights = self.prune_graphs(graphs, weights)
        return super().reset(graphs, weights)

    def _check_tensor(self, name: str, consumer: str, phase: str) -> bool:
        """Check if the tensor should be processed

        Parameters
        -------
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        phase: str
            The phase mark teacher| student| null

        Returns
        -------
        vaild: bool
            Whether to process the tensor.
        """

        # no tensor should be processed
        return False

    def prune_graphs(
        self, graphs: List[MSCGraph], weights: List[Dict[str, tvm.nd.array]]
    ) -> Tuple[List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
        """Reset the tool

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        """

        def _prune_by_shape(tensor: MSCTensor, shape: List[int]):
            return MSCTensor(tensor.name, tensor.dtype, tensor.layout.name, shape, tensor.alias)

        def _prune_by_channel(tensor: MSCTensor, dim, channel_axis: int = None):
            shape = tensor.get_shape()
            if channel_axis is None:
                channel_axis = tensor.layout_of("C")
            shape[channel_axis] = dim
            return _prune_by_shape(tensor, shape)

        new_graphs, new_weights = [], []
        pruned_weights_cnt = 0
        for graph, sub_weights in zip(graphs, weights):
            pruned_tensors, pruned_weights = {}, {}
            for node in graph.get_nodes():
                for weight in node.get_weights().values():
                    w_name = weight.name
                    if w_name in self._runtime_config:
                        data = msc_utils.cast_array(sub_weights[w_name])
                        in_axis, out_axis = self._get_io_axes(self.find_w_node(w_name))
                        w_config = self._runtime_config[w_name]
                        if w_config["in_indices"]:
                            data = prune_axis(data, in_axis, w_config["in_indices"])
                        if w_config["out_indices"]:
                            data = prune_axis(data, out_axis, w_config["out_indices"])
                        pruned_tensors[w_name] = _prune_by_shape(weight, data.shape)
                        pruned_weights[w_name] = tvm.nd.array(data)
                        pruned_weights_cnt += 1
                    else:
                        pruned_weights[w_name] = sub_weights[w_name]
                if (
                    node.optype in self._prunable_types
                    and node.weight_at("weight").name in pruned_tensors
                ):
                    out = node.output_at(0)
                    if node.optype in ("msc.linear", "msc.linear_bias"):
                        channel_axis = out.ndim - 1
                    else:
                        channel_axis = out.layout_of("C")
                    pruned_tensors[out.name] = _prune_by_channel(
                        out,
                        pruned_tensors[node.weight_at("weight").name].dim_at("O"),
                        channel_axis,
                    )
                else:
                    for out in node.get_outputs():
                        if out.name in self._runtime_config:
                            pruned_tensors[out.name] = _prune_by_channel(
                                out, len(self._runtime_config[out.name]["out_indices"])
                            )
                        elif (
                            node.get_inputs()
                            and node.input_at(0).name in pruned_tensors
                            and node.input_at(0).layout_of("C") >= 0
                            and out.layout_of("C") >= 0
                        ):
                            pruned_tensors[out.name] = _prune_by_channel(
                                out, pruned_tensors[node.input_at(0).name].dim_at("C")
                            )
            pruned_graph = _ffi_api.PruneWeights(graph, pruned_tensors)
            if MSCMap.get(MSCKey.ON_DEBUG, False):
                pruned_graph.visualize(
                    msc_utils.get_debug_dir().relpath(pruned_graph.name + "_pruned.prototxt")
                )
            new_graphs.append(pruned_graph)
            new_weights.append(pruned_weights)
        # log compress rate
        def _flatten_size(weights):
            weight_size = 0
            for sub_weights in weights:
                for w in sub_weights.values():
                    weight_size += w.asnumpy().size
            return weight_size

        raw_size = _flatten_size(weights)
        new_size = _flatten_size(new_weights)
        self._logger.info(
            "Pruned {} weights to {:g}%".format(pruned_weights_cnt, new_size * 100 / raw_size)
        )
        return new_graphs, new_weights

    def create_runtime_config(self) -> dict:
        """Create runtime config by strategy

        Returns
        -------
        runtime_config: dict
           The runtime config
        """

        rt_config = {}

        def _get_in_indices(w_node: WeightJoint) -> List[int]:
            """Get input indices for weight node"""
            if not w_node.parents:
                return []
            if w_node.name in rt_config and "in_indices" in rt_config[w_node.name]:
                return rt_config[w_node.name]["in_indices"]
            assert all(
                p.name in rt_config for p in w_node.parents
            ), "Missing some parents in runtime config " + str(w_node)
            if len(w_node.parents) == 1:
                return rt_config[w_node.parents[0].name]["out_indices"]
            if w_node.parents[0].friends:
                return rt_config[w_node.parents[0].friends[0].name]["out_indices"]
            raise Exception("Unexpected w_node " + str(w_node))

        def _prunable(w_node: WeightJoint) -> bool:
            """Check if weight node is prunable"""
            if w_node.get_attr("prune_strategy") != "prune":
                return False
            if not w_node.children:
                return False
            childrens = list(w_node.children)
            while childrens:
                current = childrens.pop(0)
                prune_strategy = current.get_attr("prune_strategy")
                if prune_strategy == "prune":
                    return True
                childrens.extend(list(current.children))
            return False

        for w_node in self.get_w_nodes():
            in_indices = _get_in_indices(w_node)
            in_axis, out_axis = self._get_io_axes(w_node)
            rt_config[w_node.name] = {"in_indices": in_indices}
            if w_node.friends and w_node != w_node.friends[0]:
                rt_config[w_node.name]["out_indices"] = rt_config[w_node.friends[0].name][
                    "out_indices"
                ]
            elif _prunable(w_node):
                method_config = self._get_method_config(w_node.name)
                method = self._get_method(w_node.name)
                rt_config[w_node.name] = method(
                    self,
                    name=w_node.name,
                    data=self.get_data(w_node.name),
                    in_axis=in_axis,
                    out_axis=out_axis,
                    in_indices=in_indices,
                    **method_config,
                )
            elif w_node.get_attr("prune_strategy") == "follow":
                rt_config[w_node.name]["out_indices"] = []
            elif w_node.get_attr("prune_strategy") == "passby":
                rt_config[w_node.name]["out_indices"] = in_indices
            else:
                rt_config[w_node.name]["out_indices"] = []
        rt_config = {n: c for n, c in rt_config.items() if c["in_indices"] or c["out_indices"]}
        self._logger.info("Config prune for {} weights".format(len(rt_config)))
        self._runtime_config = rt_config
        return self._runtime_config

    def get_w_nodes(self) -> Iterable[WeightJoint]:
        """Get all the weight nodes in the weight_graphs.

        Returns
        -------
        nodes: generator<WeightJoint>
            The generator of weight nodes.
        """

        for g in self._weight_graphs:
            for n in g.get_nodes():
                yield n

    def find_w_node(self, name: str) -> WeightJoint:
        """Find weight node by name.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        node: WeightJoint
            The found node.
        """

        for g in self._weight_graphs:
            if g.has_node(name):
                return g.find_node(name)
        raise Exception("Can not find node {} from graphs".format(name))

    def _get_io_axes(self, w_node: WeightJoint) -> Tuple[int, int]:
        """Get the input output axes

        Parameters
        ----------
        w_node: WeightJoint
            The weight node.

        Returns
        -------
        axes: (int, int)
            The input output axis.
        """

        if w_node.weight.ndim == 1:
            return 0, 0
        if w_node.has_attr("in_axis") and w_node.has_attr("out_axis"):
            return int(w_node.get_attr("in_axis")), int(w_node.get_attr("out_axis"))
        return w_node.weight.layout_of("I"), w_node.weight.layout_of("O")

    @classmethod
    def tool_type(cls):
        return MSCToolType.PRUNE


class DefaultPruner(MSCPruner):
    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultPruner)
