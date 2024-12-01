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
"""tvm.contrib.msc.core.frontend.trace.graph"""

from functools import partial
from typing import List, Any, Dict, Tuple, Iterable, Union
import numpy as np

from tvm.contrib.msc.core import utils as msc_utils
from .utils import trace_node


class TracedTensor(object):
    """Tensor object for tracing

    Parameters
    ----------
    name: str
        The name of the tensor.
    shape: list<int>
        The shape of the tensor.
    dtype: str
        The dtype of the tensor.
    layout: str
        The layout of the tensor.
    device: str
        The device of tensor.
    module: str
        The module name of the tensor.
    data:
        The data of the tensor.
    alias: str
        The alias.
    """

    traced_attrs = {}

    def __init__(
        self,
        name: str,
        shape: List[int],
        dtype: str,
        device: str = None,
        module: str = None,
        layout: str = None,
        data: Any = None,
        alias: str = None,
    ):
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._module = module
        self._layout = layout
        self._data = data
        self._alias = alias

    def __str__(self):
        return "{}{}[{},{}{}] @ {}.{}".format(
            self._name,
            "({})".format(self._alias) if self._alias else "",
            ";".join([str(s) for s in self._shape]) if self._shape else "()",
            self._dtype,
            "" if not self._layout else "," + self._layout,
            self._module or "any",
            self._device or "any",
        )

    def __getattr__(self, name):
        if name in self.traced_attrs:
            return partial(self._traced_attr, attr_name=name)
        if hasattr(self._data, name):
            return getattr(self._data, name)
        raise Exception("Can not find attribute {} form data {}".format(name, self._data))

    def __add__(self, other):
        output = self._data.__add__(self.to_data(other))
        return self._trace_operator("add", [self, other], [output])

    def __sub__(self, other):
        output = self._data.__sub__(self.to_data(other))
        return self._trace_operator("sub", [self, other], [output])

    def __mul__(self, other):
        output = self._data.__mul__(self.to_data(other))
        return self._trace_operator("mul", [self, other], [output])

    def __truediv__(self, other):
        output = self._data.__truediv__(self.to_data(other))
        return self._trace_operator("truediv", [self, other], [output])

    def __pow__(self, other):
        output = self._data.__pow__(self.to_data(other))
        return self._trace_operator("pow", [self, other], [output])

    def __eq__(self, other):
        output = self._data.__eq__(self.to_data(other))
        return self._trace_operator("eq", [self, other], [output])

    def __ne__(self, other):
        output = self._data.__ne__(self.to_data(other))
        return self._trace_operator("ne", [self, other], [output])

    def __le__(self, other):
        output = self._data.__le__(self.to_data(other))
        return self._trace_operator("le", [self, other], [output])

    def __lt__(self, other):
        output = self._data.__lt__(self.to_data(other))
        return self._trace_operator("lt", [self, other], [output])

    def __ge__(self, other):
        output = self._data.__ge__(self.to_data(other))
        return self._trace_operator("ge", [self, other], [output])

    def __gt__(self, other):
        output = self._data.__gt__(self.to_data(other))
        return self._trace_operator("gt", [self, other], [output])

    def __getitem__(self, key):
        v_key, inputs = self._get_slice(key)
        output = self._data.__getitem__(v_key)
        return self._trace_operator("getitem", [self] + inputs, [output], {"slice": key})

    def __setitem__(self, key, value):
        v_key, inputs = self._get_slice(key)
        self._data.__setitem__(v_key, self.to_data(value))
        return self._trace_operator("setitem", [self, value] + inputs, [self._data], {"slice": key})

    def _get_slice(self, slice_obj: List[Any]) -> Tuple[List[Any], List["TracedTensor"]]:
        """Get the slice attribute

        Parameters
        ----------
        slice_obj:
            The slice attribute.

        Returns
        -------
        slice: int
            The valid slice.
        inputs: list<TracedTensor>
            The inputs tensors.
        """

        inputs = []
        if isinstance(slice_obj, tuple):
            v_slice = tuple([self.to_data(s) for s in slice_obj])
            inputs.extend(k for k in slice_obj if isinstance(k, TracedTensor))
        else:
            v_slice = self.to_data(slice_obj)
            if isinstance(slice_obj, TracedTensor):
                inputs.append(slice_obj)
        return v_slice, inputs

    def _traced_attr(self, *args, attr_name: str = None, **kwargs):
        assert attr_name in self.traced_attrs, "Can not find attr {} in traced_attrs".format(
            attr_name
        )
        for info in self.traced_attrs[attr_name]:
            if info["checker"](*args, **kwargs):
                return info["func"](self, *args, **kwargs)
        raise Exception("Can not match arg {}, kwargs {} for {}".format(args, kwargs, attr_name))

    def _trace_operator(
        self, optype: str, inputs: List[Any], outputs: List[Any], attrs: Dict[str, Any] = None
    ) -> Union[List["TracedTensor"], "TracedTensor"]:
        """Trace the operator and gte outputs

        Parameters
        ----------
        optype: str
            The type of node.
        inputs: list<any>
            The input datas.
        outputs: list<any>
            The output datas.
        attrs: dict<string, any>
            The attributes of the node.

        Returns
        -------
        outputs: list<TracedTensor>/TracedTensor
            The outputs.
        """

        outputs = [{"obj": o, "module": self._module} for o in outputs]
        node = trace_node(optype, inputs, outputs, attrs=attrs)
        if attrs and attrs.get("multi_outputs", False):
            return node.get_ouputs()
        return node.output_at(0)

    def set_alias(self, alias: str):
        """Set alias for the tensor

        Parameters
        ----------
        alias: str
            The alias.
        """

        self._alias = alias

    def inspect(self):
        """Inspect the tensor"""

        des = str(self)
        if self._data is not None:
            info = msc_utils.inspect_array(self._data, as_str=False)
            des += ": Max {:g}, Min {:g}, Avg {:g}".format(info["max"], info["min"], info["avg"])
        return des

    @property
    def name(self):
        return self._name

    @property
    def alias(self):
        return self._alias

    @property
    def shape(self):
        if "shape" in self.traced_attrs and hasattr(self._data, "shape"):
            return self._trace_operator(
                "shape", [self], [np.array(self._data.shape, dtype="int32")]
            )
        return self._shape

    @property
    def dtype(self):
        if "dtype" in self.traced_attrs and hasattr(self._data, "dtype"):
            return self._data.dtype
        return self._dtype

    @property
    def device(self):
        if "device" in self.traced_attrs and hasattr(self._data, "device"):
            return self._data.device
        return self._device

    @property
    def module(self):
        return self._module

    @property
    def data(self):
        return self._data

    @classmethod
    def to_tensor(cls, obj: Any, name: str, **kwargs):
        """Create TracedTensor from tensor"""

        try:
            import torch  # pylint: disable=import-outside-toplevel

            is_torch = isinstance(obj, torch.Tensor)
        except:  # pylint: disable=bare-except
            is_torch = False

        kwargs["data"] = obj
        if isinstance(obj, str):
            kwargs = msc_utils.update_dict({"shape": ["seq:0:4096"], "dtype": "string"}, kwargs)
        elif isinstance(obj, (int, np.int32)):
            kwargs = msc_utils.update_dict({"shape": [], "dtype": "int32"}, kwargs)
        elif isinstance(obj, (np.int64)):
            kwargs = msc_utils.update_dict({"shape": [], "dtype": "int64"}, kwargs)
        elif isinstance(obj, (float, np.float32)):
            kwargs = msc_utils.update_dict({"shape": [], "dtype": "float32"}, kwargs)
        elif isinstance(obj, (bool, np.bool_)):
            kwargs = msc_utils.update_dict({"shape": [], "dtype": "bool"}, kwargs)
        elif isinstance(obj, np.ndarray):
            kwargs = msc_utils.update_dict(
                {
                    "shape": obj.shape,
                    "dtype": obj.dtype.name,
                    "device": "cpu",
                    "module": "numpy",
                },
                kwargs,
            )
        elif is_torch:
            ref_obj = msc_utils.MSCArray(obj)
            array = ref_obj._to_ndarray()
            kwargs = msc_utils.update_dict(
                {
                    "shape": array.shape,
                    "dtype": array.dtype.name,
                    "device": ref_obj.device,
                    "module": "torch",
                },
                kwargs,
            )
        else:
            raise Exception("Unexpected object {}({})".format(obj, type(obj)))
        return cls(name, **kwargs)

    @classmethod
    def to_data(cls, obj: Any):
        """Get data from tensor"""

        if isinstance(obj, cls):
            return obj.data
        return obj


class TracedNode(object):
    """Node object for tracing

    Parameters
    ----------
    index: int
        The index of the node
    name: str
        The name of the node.
    scope: str
        The scope of the node.
    optype: str
        The type of node.
    attrs: dict<string, any>
        The attributes of the node.
    inputs: list<tuple<TracedNode, int>>
        The inputs of the node in format <parent,out_idx>.
    outputs: list<TracedTensor>
        The outputs of the node.
    meta:
        The meta operator
    """

    def __init__(
        self,
        index: int,
        name: str,
        scope: str,
        optype: str,
        inputs: List[Tuple["TracedNode", int]],
        outputs: List[TracedTensor],
        attrs: Dict[str, Any] = None,
        meta: Any = None,
    ):
        self._index = index
        self._name = name
        self._scope = scope
        self._optype = optype
        self._attrs = attrs or {}
        self._inputs = []
        self._outputs = outputs
        self._parents = []
        self._children = []
        for i in inputs:
            self.add_input(i)
        self._meta = meta

    def __str__(self):
        info = "N_{} {}{}<P: {}| C: {}>\n  OP: {}".format(
            self._index,
            self._name,
            "({})".format(self._scope) if self._scope else "",
            ";".join([p.name for p in self._parents]),
            ";".join([c.name for c in self._children]),
            self._optype,
        )
        if self._inputs:
            info += "\n  IN: " + ";".join([str(self.input_at(i)) for i in range(len(self._inputs))])
        info += "\n  OUT: " + ";".join([str(o) for o in self._outputs])
        if self._attrs:
            info += "\n  ATTRS: " + ";".join(["{}={}".format(k, v) for k, v in self._attrs.items()])
        if self._meta:
            info += "\n  META: " + str(type(self._meta))
        return info

    def add_parent(self, parent: "TracedNode") -> int:
        """Add parent for the node

        Parameters
        ----------
        parent: TracedNode
            The parent node.

        Returns
        -------
        index: int
            The parent index.
        """

        for idx, p in enumerate(self._parents):
            if p.name == parent.name:
                return idx
        self._parents.append(parent)
        parent.add_child(self)
        return len(self._parents) - 1

    def add_child(self, child: "TracedNode") -> int:
        """Add child for the node

        Parameters
        ----------
        child: TracedNode
            The child node.

        Returns
        -------
        index: int
            The child index.
        """

        for idx, c in enumerate(self._children):
            if c.name == child.name:
                return idx
        self._children.append(child)
        child.add_parent(self)
        return len(self._children) - 1

    def add_input(self, input: Tuple["TracedNode", int]) -> int:
        """Add input for the node

        Parameters
        ----------
        input: tuple<TracedNode, int>
            The input reference.

        Returns
        -------
        index: int
            The input index.
        """

        self._inputs.append(input)
        self.add_parent(input[0])
        return len(self._inputs) - 1

    def input_at(self, idx: int) -> TracedTensor:
        """Get input by index

        Parameters
        ----------
        idx: int
            The input index.

        Returns
        -------
        input: TracedTensor
            The input.
        """

        parent, o_idx = self._inputs[idx]
        return parent.output_at(o_idx)

    def get_inputs(self) -> List[TracedTensor]:
        """Get inputs of node

        Returns
        -------
        inputs: list<TracedTensor>
            The inputs.
        """

        return [self.input_at(i) for i in range(len(self._inputs))]

    def output_at(self, idx: int) -> TracedTensor:
        """Get output by index

        Parameters
        ----------
        idx: int
            The output index.

        Returns
        -------
        output: TracedTensor
            The output.
        """

        return self._outputs[idx]

    def get_outputs(self) -> List[TracedTensor]:
        """Get outputs of node

        Returns
        -------
        outputs: list<TracedTensor>
            The outputs.
        """

        return self._outputs

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Get the attr by key

        Parameters
        ----------
        key: str
            The attr key.
        default:
            The default value.

        Returns
        -------
        attr:
            The attr value.
        """

        return self._attrs.get(key, default)

    def get_arg_attr(self, idx: int, default: Any = None) -> Any:
        """Get the argument attr by index

        Parameters
        ----------
        idx: int
            The index of arg attrs.
        default:
            The default value.

        Returns
        -------
        attr:
            The attr value.
        """

        args = self.get_attr("args")
        if not args or len(args) <= idx:
            return default
        return args[idx]

    def infer_module(self) -> str:
        """Inference the module of node

        Returns
        -------
        module: str
            The module of the node.
        """

        module = None
        for i in self.get_inputs() + self.get_outputs():
            if not i.module:
                continue
            if not module:
                module = i.module
            assert module == i.module, "Find different modules of inputs: " + str(self)
        assert module, "Can not infer module of " + str(self)
        return module

    @property
    def name(self):
        return self._name

    @property
    def optype(self):
        return self._optype

    @property
    def attrs(self):
        return self._attrs

    @property
    def parents(self):
        return self._parents

    @property
    def children(self):
        return self._children

    @property
    def meta(self):
        return self._meta


class TracedGraph(object):
    """Graph object for tracing

    Parameters
    ----------
    name: str
        The name of the graph.
    """

    def __init__(self, name: str):
        self._name = name
        self._nodes, self._node_names = {}, []
        self._scopes, self._scope_names = {}, []
        self._current_scope = None
        self._inputs, self._outputs = [], []
        self._tensor_alias = {}

    def __str__(self):
        info = "{} <IN: {}| OUT: {}>\n".format(
            self._name, ";".join(self._inputs), ";".join(self._outputs)
        )
        info += "\n\n".join([str(n) for n in self.get_nodes()])
        return info

    def new_scope(self, name: str = None):
        """Start new scope for the graph

        Parameters
        ----------
        name: str
            The name of the scope.
        """

        name = name or "group_" + str(len(self._scopes))
        self._current_scope = name
        self._scopes[name] = []
        self._scope_names.append(name)

    def add_node(
        self,
        optype: str,
        inputs: List[Any],
        outputs: List[Any],
        name: str = None,
        attrs: Dict[str, Any] = None,
        meta: Any = None,
    ) -> TracedNode:
        """Add node in graph

        Parameters
        ----------
        optype: str
            The type of node.
        inputs: list<any>
            The input datas.
        outputs: list<any>
            The output datas.
        name: str
            The name of the node.
        attrs: dict<string, any>
            The attributes of the node.
        meta:
            The meta operator.

        Returns
        -------
        node: TracedNode
            The node.
        """

        v_inputs, v_outputs = [], []
        for idx, i_data in enumerate(inputs):
            if isinstance(i_data, TracedTensor):
                p_name, o_idx = i_data.name.split(":")
                v_inputs.append((self.find_node(p_name), int(o_idx)))
            else:
                c_attrs = {"scalar": i_data} if isinstance(i_data, (int, float)) else None
                kwargs = {"obj": i_data}
                for i in inputs:
                    if msc_utils.is_array(i):
                        i_array = msc_utils.MSCArray(i_data)
                        kwargs.update(
                            {
                                "dtype": i_array.cast().dtype.name,
                                "device": i_array.device,
                                "module": i_array.module,
                            }
                        )
                        break
                    elif isinstance(i, TracedTensor):
                        kwargs.update({"dtype": i.dtype, "device": i.device, "module": i.module})
                        break
                c_node = self.add_node("constant", [], [kwargs], attrs=c_attrs)
                v_inputs.append((c_node, 0))
        name = name or "node_" + str(len(self._nodes))
        for idx, o_data in enumerate(outputs):
            if not isinstance(o_data, dict):
                o_data = {"obj": o_data}
            o_data["name"] = "{}:{}".format(name, idx)
            v_outputs.append(TracedTensor.to_tensor(**o_data))
        node = TracedNode(
            len(self._nodes), name, self._current_scope, optype, v_inputs, v_outputs, attrs, meta
        )
        self._nodes[name] = node
        self._node_names.append(name)
        if self._current_scope:
            self._scopes[self._current_scope].append(name)
        if optype == "input":
            self._inputs.append(node.output_at(0).name)
        return node

    def finalize(self, outputs: List[str]):
        """Finalize the graph

        Parameters
        ----------
        outputs: list<str>
            The output names of the graph.
        """

        self._outputs = outputs
        for n in self.get_nodes():
            for o in n.get_outputs():
                if o.alias:
                    self._tensor_alias[o.alias] = o.name

    def group_up(self) -> dict:
        """Extract group info

        Returns
        -------
        info: dict
            The groups info.
        """

        groups = []
        for name in self._scope_names:
            nodes_set = set(self._scopes[name])
            g_inputs, g_outputs = [], []
            for n in self._scopes[name]:
                node = self.find_node(n)
                for i in node.get_inputs():
                    if self.find_producer(i).name not in nodes_set and i.name not in g_inputs:
                        g_inputs.append(i.name)
                for o in node.get_outputs():
                    consumers = self.find_consumers(o)
                    if not consumers and o.name not in g_outputs:
                        g_outputs.append(o.name)
                    elif (
                        any(c.name not in nodes_set for c in consumers) and o.name not in g_outputs
                    ):
                        g_outputs.append(o.name)
            group = {
                "name": name,
                "inputs": g_inputs,
                "outputs": g_outputs,
                "nodes": self._scopes[name],
            }
            groups.append(group)
        return {"inputs": self._inputs, "outputs": self._outputs, "groups": groups}

    def find_node(self, name: str) -> TracedNode:
        """Find the node by name

        Parameters
        ----------
        name: str
            The name of the node.

        Returns
        -------
        node: TracedNode
            The node.
        """

        assert name in self._nodes, "Can not find node " + str(name)
        return self._nodes[name]

    def get_nodes(self) -> Iterable[TracedNode]:
        """Get all nodes in graph

        Returns
        -------
        nodes: list<TracedNode>
            The nodes.
        """

        for n in self._node_names:
            yield self._nodes[n]

    def find_producer(self, tensor: TracedTensor) -> TracedNode:
        """Find the producer of tensor

        Parameters
        ----------
        tensor: TracedTensor
            The tensor.

        Returns
        -------
        producer: TracedNode
            The producer.
        """

        p_name, _ = tensor.name.split(":")
        return self.find_node(p_name)

    def find_consumers(self, tensor: TracedTensor) -> List[TracedNode]:
        """Find the consumers of tensor

        Parameters
        ----------
        tensor: TracedTensor
            The tensor.

        Returns
        -------
        consumers: list<TracedNode>
            The consumers.
        """

        consumers = []
        for c in self.find_producer(tensor).children:
            c_inputs = set([i.name for i in c.get_inputs()])
            if tensor.name in c_inputs:
                consumers.append(c)
        return consumers

    def find_tensor(self, t_name: str) -> TracedTensor:
        """Find the tensor by name

        Parameters
        ----------
        t_name: str
            The tensor name.

        Returns
        -------
        tensor: TracedTensor
            The tensor.
        """

        t_name = self._tensor_alias.get(t_name, t_name)
        p_name, o_idx = t_name.split(":")
        return self.find_node(p_name).output_at(int(o_idx))

    def inspect(self) -> dict:
        """Extract important info of the graph.

        Returns
        -------
        graph_des: dict
            The graph description in json format.
        """

        graph_des = {
            "inputs": self._inputs,
            "outputs": self._outputs,
            "scopes": len(self._scopes),
            "nodes": {"total": 0},
        }
        for node in self.get_nodes():
            graph_des["nodes"].setdefault(node.optype, 0)
            graph_des["nodes"]["total"] += 1
            graph_des["nodes"][node.optype] += 1
        return graph_des

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs
