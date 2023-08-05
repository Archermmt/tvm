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
"""tvm.contrib.msc.core.ir.graph"""

from typing import Dict, Tuple, List, Optional, Union
import numpy as np

import tvm
from tvm.runtime import Object
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils


@tvm._ffi.register_object("msc.core.MSCTensor")
class MSCTensor(Object):
    """Tensor in MSCGraph

    Parameters
    ----------
    name: string
        The name of the tensor.
    dtype: string or np.dtype or DataType
        The data type the tensor.
    layout: string
        The layout of the tensor.
    shape: list<int>
        The shape of the tensor.
    alias: string
        The alias of the tensor.
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        layout: Union[str, np.dtype, tvm.DataType],
        shape: List[int],
        alias: Optional[str] = None,
    ):
        if not isinstance(dtype, tvm.DataType):
            dtype = tvm.DataType(dtype)
        self.__init_handle_by_constructor__(
            _ffi_api.MSCTensor, name, dtype, layout, shape, alias or ""
        )

    def get_shape(self):
        return [int(i) for i in self.shape]

    @property
    def dtype_name(self):
        return _ffi_api.MSCTensorDTypeName(self)

    @property
    def ndim(self):
        return len(self.shape)


class BaseJoint(Object):
    """Base class of all MSC Nodes."""


@tvm._ffi.register_object("msc.core.MSCJoint")
class MSCJoint(BaseJoint):
    """Node in MSCGraph

    Parameters
    ----------
    index: int
        The index of the node.
    name: string
        The name of the node.
    master: string
        The master of the node.
    optype: string
        The optype of the node.
    attrs: dict<string, string>
        The attributes of the node.
    inputs: list<tuple<MSCJoint, int>>
        The inputs of the node in format <parent,out_idx>.
    outputs: list<MSCTensor>
        The outputs of the node.
    weights: dict<string, MSCTensor>
        The weights of the node.
    """

    def __init__(
        self,
        index: int,
        name: str,
        master: str,
        optype: str,
        attrs: Dict[str, str],
        inputs: List[Tuple[BaseJoint, int]],
        outputs: List[MSCTensor],
        weights: Dict[str, MSCTensor],
    ):

        parents = [i[0] for i in inputs]
        out_indices = [i[1] for i in inputs]
        self.__init_handle_by_constructor__(
            _ffi_api.MSCJoint,
            index,
            name,
            master,
            optype,
            attrs,
            parents,
            out_indices,
            outputs,
            weights,
        )

    def input_at(self, idx: int) -> MSCTensor:
        """Get input at idx.

        Parameters
        ----------
        idx: int
            The index of input.

        Returns
        -------
        input: MSCTensor
            The input Tensor.
        """

        return _ffi_api.MSCJointInputAt(self, idx)

    def output_at(self, idx: int) -> MSCTensor:
        """Get output at idx.

        Parameters
        ----------
        idx: int
            The index of output.

        Returns
        -------
        output: MSCTensor
            The output Tensor.
        """

        return _ffi_api.MSCJointOutputAt(self, idx)

    def get_inputs(self) -> List[MSCTensor]:
        """Get all the inputs.

        Returns
        -------
        inputs: list<MSCJoint>
            The input Tensors.
        """

        return _ffi_api.MSCJointGetInputs(self)

    def get_outputs(self) -> List[MSCTensor]:
        """Get all the outputs.

        Returns
        -------
        outputs: list<MSCJoint>
            The output Tensors.
        """

        return _ffi_api.MSCJointGetOutputs(self)


@tvm._ffi.register_object("msc.core.WeightJoint")
class WeightJoint(BaseJoint):
    """Node in WeightGraph

    Parameters
    ----------
    index: int
        The index of the node.
    name: string
        The name of the node.
    master: string
        The master of the node.
    optype: string
        The optype of the node.
    wtype: string
        The weight type of the node.
    attrs: dict<string, string>
        The attributes of the node.
    weight: MSCTensor,
        The weight of the node.
    parents: list<WeightJoint>
        The parents of the node.
    friends: list<WeightJoint>
        The friends of the node.
    """

    def __init__(
        self,
        index: int,
        name: str,
        master: str,
        optype: str,
        wtype: str,
        attrs: Dict[str, str],
        weight: MSCTensor,
        parents: List[BaseJoint],
        friends: List[BaseJoint],
    ):

        self.__init_handle_by_constructor__(
            _ffi_api.WeightJoint,
            index,
            name,
            master,
            optype,
            wtype,
            attrs,
            weight,
            parents,
            friends,
        )


class BaseGraph(Object):
    """Base class of all MSC Graphs."""


@tvm._ffi.register_object("msc.core.MSCGraph")
class MSCGraph(BaseGraph):
    """The MSCGraph

    Parameters
    ----------
    name: string
        The name of the graph.
    nodes: list<MSCJoint>
        The nodes of the graph.
    input_names: list<str>
        The input names of the graph.
    output_names: list<str>
        The output names of the graph.
    """

    def __init__(
        self,
        name: str,
        nodes: List[MSCJoint],
        input_names: List[str],
        output_names: List[str],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.MSCGraph,
            name,
            nodes,
            input_names,
            output_names,
        )

    def find_node(self, name: str) -> MSCJoint:
        """Find node by name.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        node: MSCJoint
            The found node.
        """

        return _ffi_api.MSCGraphFindNode(self, name)

    def find_tensor(self, name: str) -> MSCTensor:
        """Find tensor by name.

        Parameters
        ----------
        name: string
            The name of the tensor.

        Returns
        -------
        node: MSCTensor
            The found tensor.
        """

        return _ffi_api.MSCGraphFindTensor(self, name)

    def find_producer(self, name: str) -> MSCJoint:
        """Find producer by tensor_name.

        Parameters
        ----------
        name: string
            The name of the tensor.

        Returns
        -------
        node: MSCJoint
            The found prducer.
        """

        return _ffi_api.MSCGraphFindProducer(self, name)

    def find_consumers(self, name: str) -> List[MSCJoint]:
        """Find consumers by tensor_name.

        Parameters
        ----------
        name: string
            The name of the tensor.

        Returns
        -------
        node: list<MSCJoint>
            The found consumers.
        """

        return _ffi_api.MSCGraphFindConsumers(self, name)

    def get_nodes(self):
        """Get all the nodes in the graph.

        Returns
        -------
        nodes: generator<MSCJoint>
            The generator of nodes.
        """

        for n in self.node_names:
            yield self.find_node(n)

    def input_at(self, idx: int) -> MSCTensor:
        """Get input at idx.

        Parameters
        ----------
        idx: int
            The index of input.

        Returns
        -------
        input: MSCTensor
            The input Tensor.
        """

        return _ffi_api.MSCGraphInputAt(self, idx)

    def output_at(self, idx: int) -> MSCTensor:
        """Get output at idx.

        Parameters
        ----------
        idx: int
            The index of output.

        Returns
        -------
        output: MSCTensor
            The output Tensor.
        """

        return _ffi_api.MSCGraphOutputAt(self, idx)

    def get_inputs(self) -> List[MSCTensor]:
        """Get all the inputs.

        Returns
        -------
        inputs: list<MSCJoint>
            The input Tensors.
        """

        return _ffi_api.MSCGraphGetInputs(self)

    def get_outputs(self) -> List[MSCTensor]:
        """Get all the outputs.

        Returns
        -------
        outputs: list<MSCJoint>
            The output Tensors.
        """

        return _ffi_api.MSCGraphGetOutputs(self)

    def to_json(self, path: Optional[str] = None) -> str:
        """Dump the graph to json.

        Parameters
        ----------
        path: string
            The file_path for save json.

        Returns
        -------
        graph_json: string
            The graph in json format.
        """

        graph_json = _ffi_api.MSCGraphToJson(self)
        if path:
            with open(path, "w") as f:
                f.write(graph_json)
        return graph_json

    @classmethod
    def from_json(cls, json_str: str) -> BaseGraph:
        """Load the graph from json.

        Parameters
        ----------
        json_str: string
            The file_path or json string.

        Returns
        -------
        graph: MSCgraph
            The graph.
        """

        dict_obj = msc_utils.load_dict(json_str)
        return _ffi_api.MSCGraphFromJson(msc_utils.dump_dict(dict_obj))

    def clone(self) -> BaseGraph:
        """Clone the graph.

        Returns
        -------
        new_graph: MSCGraph
            The cloned graph.
        """

        return MSCGraph.from_json(self.to_json())

    def is_same(self, other: BaseGraph) -> bool:
        """A fast method to check if two graphs are same.

        Returns
        -------
        other: MSCGraph
            The graph to be compared.

        Returns
        -------
        is_same: bool
            Whether two graphs are the same.
        """

        if self.node_names != other.node_names:
            return False
        if self.input_names != other.input_names or self.output_names != other.output_names:
            return False
        for s_i, o_i in zip(self.get_inputs(), other.get_inputs()):
            if not s_i.is_same(o_i):
                return False
        for s_o, o_o in zip(self.get_outputs(), other.get_outputs()):
            if not s_o.is_same(o_o):
                return False
        for s_n, o_n in zip(self.get_nodes(), other.get_nodes()):
            if not s_n.is_same(o_n):
                return False
        return True

    def visualize(self, path: Optional[str] = None) -> str:
        """Dump the graph to prototxt format.

        Parameters
        ----------
        path: string
            The file_path to save prototxt.

        Returns
        -------
        graph_proto: string
            The graph in prototxt format.
        """

        graph_proto = _ffi_api.MSCGraphToPrototxt(self)
        if path:
            with open(path, "w") as f:
                f.write(graph_proto)
        return graph_proto


@tvm._ffi.register_object("msc.core.WeightGraph")
class WeightGraph(Object):
    """The WeightGraph

    Parameters
    ----------
    name: string
        The name of the graph.
    nodes: list<WeightJoint>
        The nodes of the graph.
    """

    def __init__(
        self,
        name: str,
        nodes: List[WeightJoint],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.WeightGraph,
            name,
            nodes,
        )
