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
"""tvm.contrib.msc.core.trace.parse.base_parser"""

import logging
from functools import partial
from typing import Any, Dict, Callable

from tvm import relax
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core.utils.register import MSCRegistery
from ..graph import *
from ..utils import register_trace_func


class BaseParser(object):
    """Parser for tracing

    Parameters
    ----------
    tracer: str
        The tracer.
    debug_level: int
        The debug level.
    logger: logging.Logger
        The logger.
    """

    def __init__(
        self,
        tracer: "Tracer",
        config: dict = None,
        debug_level: int = 0,
        logger: logging.Logger = None,
    ):
        self._tracer = tracer
        self._config = config or {}
        self._debug_level = debug_level
        self._logger = logger or msc_utils.get_global_logger()
        self._logger.info(msc_utils.msg_block(self.mark("SETUP"), self.setup()))

    def setup(self):
        """Setup the tracer"""

        self._convert_map = self._create_convert_map()
        self._traced_funcs, self._tensor_attrs = {}, {}
        return {"converters": len(self._convert_map), "debug_level": self._debug_level}

    def convert(self, node: TracedNode) -> relax.Expr:
        """Convert node to relax Expr"""

        assert node.optype in self._convert_map, "Convert is not support for {}".format(node.optype)
        expr = self._convert_map[node.optype](node)
        if isinstance(expr, (relax.Var, relax.Constant, relax.ShapeExpr)):
            return expr
        return self.emit(expr, node.name)

    def retrieve_arg(self, tensor: TracedTensor) -> relax.Var:
        """Retrieve argument from tensor

        Parameters
        -------
        tensor: TracedTensor
            The traced tensor.

        Returns
        -------
        argument: relax.Expr
            The argument.
        """

        return self._tracer.env[tensor.name]

    def retrieve_args(self, node: TracedNode) -> List[relax.Var]:
        """Retrieve arguments of node

        Parameters
        -------
        node: TracedNode
            The traced node.

        Returns
        -------
        arguments: list<relax.Expr>
            The arguments.
        """

        return [self.retrieve_arg(i) for i in node.get_inputs()]

    def emit(self, expr: relax.Expr, name_hint: str = None) -> relax.Var:
        """Emit the expr to var

        Parameters
        -------
        expr: relax.Expr
            The relax expr.
        name_hint: str
            The name hint of expr.

        Returns
        -------
        var: relax.Var
            The emitted var.
        """

        if name_hint:
            return self._tracer.block_builder.emit(expr, name_hint)
        return self._tracer.block_builder.emit(expr)

    def check_args_module(self, *args, module: str = None, **kwargs) -> bool:
        """Infer the module of arguments

        Parameters
        -------
        module: str
            The module name.
        args:
            The arguments.
        kwargs:
            The kwargs.

        Returns
        -------
        fit: bool
            Whether all arguments belongs to the module.
        """

        for arg in args:
            if isinstance(arg, TracedTensor) and arg.module != module:
                return False
        for value in kwargs.values():
            if isinstance(value, TracedTensor) and value.module != module:
                return False
        return True

    def _unary_op(self, node: TracedNode, op: Callable) -> relax.Var:
        inputs = self.retrieve_args(node)
        assert len(inputs) == 1, "unary op only support 1 inputs, get " + str(inputs)
        return self.emit(op(inputs[0]))

    def _binary_op(self, node: TracedNode, op: Callable) -> relax.Var:
        inputs = self.retrieve_args(node)
        assert len(inputs) == 2, "binary op only support 2 inputs, get " + str(inputs)
        return self.emit(op(*inputs))

    def astype(self, node: TracedNode) -> relax.Var:
        data = self.retrieve_arg(node.input_at(0))
        dtype = node.get_attr("dtype") or node.get_arg_attr(1)
        return relax.op.astype(data, str(dtype))

    def constant(self, node: TracedNode) -> relax.Var:
        scalar, output = node.get_attr("scalar"), node.output_at(0)
        if scalar is None:
            return relax.const(msc_utils.cast_array(output.data), output.dtype)
        return relax.const(scalar, output.dtype)

    def getitem(self, node: TracedNode) -> relax.Var:
        a_slice = node.get_attr("slice")
        data = self.retrieve_arg(node.input_at(0))
        if isinstance(a_slice, int):
            sliced = self.emit(
                relax.op.strided_slice(data, [0], [a_slice], [a_slice + 1], [1]),
                node.name + "." + str(a_slice),
            )
            return relax.op.reshape(sliced, list(data.struct_info.shape)[1:])
        if isinstance(a_slice, slice):
            if a_slice.start is None and a_slice.stop is None and a_slice.step == -1:
                return relax.op.flip(data, axis=0)
            stride_begin = [0 if a_slice.start is None else a_slice.start]
            stride_end = [data.struct_info.shape[0] if a_slice.stop is None else a_slice.stop]
            stride = [1 if a_slice.step is None else a_slice.step]
            return relax.op.strided_slice(data, [0], stride_begin, stride_end, stride)
        if isinstance(a_slice, TracedTensor):
            return relax.op.take(data, self.retrieve_arg(a_slice), 0)
        take_indices, take_axes = [], []
        stride_begin, stride_end, stride_axes = [], [], []
        stride = []
        expand_dim, reduce_dim = [], []
        shape, i = data.struct_info.shape, 0
        for index in a_slice:
            if isinstance(index, int):
                stride_begin.append(index)
                stride_end.append(index + 1)
                stride.append(1)
                stride_axes.append(i)
                reduce_dim.append(i)
                i = i + 1
            elif isinstance(index, slice):
                stride_begin.append(0 if index.start is None else index.start)
                stride_end.append(shape[i] if index.stop is None else index.stop)
                stride.append(1 if index.step is None else index.step)
                stride_axes.append(i)
                i = i + 1
            elif index is None:
                expand_dim.append(len(stride_axes) + len(expand_dim))
            elif isinstance(index, TracedTensor):
                node_index = self.retrieve_arg(index)
                if not isinstance(node_index, relax.Expr):
                    raise ValueError(
                        "Unsupported index type for relax.op.take: " + str(type(node_index))
                    )
                take_indices.append(node_index)
                take_axes.append(i)
                i = i + 1
            else:
                raise ValueError("Unsupported index type: " + str(type(index)))
        while i < len(shape):
            stride_begin.append(0)
            stride_end.append(shape[i])
            stride.append(1)
            stride_axes.append(i)
            i += 1
        taken = data
        if len(take_indices) > 1:
            raise ValueError("Multiple tensors as index not yet supported")
        for index, axis in zip(take_indices, take_axes):
            taken = self.emit(relax.op.take(taken, index, axis), node.name + "_take_" + str(axis))
        sliced = self.emit(
            relax.op.strided_slice(taken, stride_axes, stride_begin, stride_end, stride), node.name
        )
        sliced_shape = list(sliced.struct_info.shape)
        if not expand_dim and not reduce_dim:
            return sliced
        for i in expand_dim:
            sliced_shape.insert(i, 1)
        sliced_shape = [i for idx, i in enumerate(sliced_shape) if idx not in reduce_dim]
        return self.emit(relax.op.reshape(sliced, sliced_shape), node.name + "_reshape")

    def setitem(self, node: TracedNode) -> relax.Var:
        a_slice = node.get_attr("slice")
        data = self.retrieve_arg(node.input_at(0))
        updates = self.retrieve_arg(node.input_at(1))
        if isinstance(a_slice, TracedTensor):
            if a_slice.dtype in ("int32", "int64"):
                indices = self.retrieve_arg(a_slice)
                exp_indices = self.emit(
                    relax.op.reshape(indices, list(indices.struct_info.shape) + [1]),
                    node.name + "_indices",
                )
                return relax.op.scatter_nd(data, exp_indices, updates)
            if a_slice.dtype == "bool":
                mask = self.retrieve_arg(a_slice)
                ndim = len(mask.struct_info.shape)
                if ndim == 1:
                    index = self.emit(relax.op.cumsum(mask, 0, dtype="int32"), node.name + "_index")
                    index = self.emit(
                        relax.op.subtract(index, relax.const(1, "int32")), node.name + "_index_sub"
                    )
                    gathered_updates = self.emit(
                        relax.op.take(updates, index, axis=0), node.name + "_gather"
                    )
                else:
                    f_mask = self.emit(relax.op.reshape(mask, [-1]), node.name + "_fmask")
                    index = self.emit(
                        relax.op.cumsum(f_mask, 0, dtype="int32"), node.name + "_index"
                    )
                    index = self.emit(
                        relax.op.subtract(index, relax.const(1, "int32")), node.name + "_index_sub"
                    )
                    updates_shape = [-1] + [
                        s for idx, s in enumerate(updates.struct_info.shape) if idx >= ndim
                    ]
                    f_updates = self.emit(
                        relax.op.reshape(updates, updates_shape), node.name + "_fupdates"
                    )
                    gathered_updates = self.emit(
                        relax.op.take(f_updates, index, axis=0), node.name + "_gather"
                    )
                    gathered_updates = self.emit(
                        relax.op.reshape(gathered_updates, data.struct_info.shape)
                    )
                if ndim != len(data.struct_info.shape):
                    diff = len(data.struct_info.shape) - ndim
                    mask = self.emit(
                        relax.op.reshape(mask, list(mask.struct_info.shape) + [1] * diff),
                        node.name + "_mask_exp",
                    )
                mask = self.emit(
                    relax.op.broadcast_to(mask, data.struct_info.shape),
                    node.name + "_broadcast",
                )
                return relax.op.where(mask, gathered_updates, data)
            raise Exception("setitem with tensor only support int and bool")
        raise NotImplementedError("slice {} is not supported for setitem".format(a_slice))

    def shape(self, node: TracedNode) -> relax.Var:
        data = self.retrieve_arg(node.input_at(0))
        return data.struct_info.shape

    def _create_convert_map(self) -> Dict[str, Callable[[TracedNode], relax.Var]]:
        """Create convert map for nodes

        Returns
        -------
        convert_map: list<str, callable>
            The convert_map.
        """

        return {
            # binary ops
            "add": partial(self._binary_op, op=relax.op.add),
            "eq": partial(self._binary_op, op=relax.op.equal),
            "ge": partial(self._binary_op, op=relax.op.greater_equal),
            "gt": partial(self._binary_op, op=relax.op.greater),
            "floordiv": partial(self._binary_op, op=relax.op.floor_divide),
            "le": partial(self._binary_op, op=relax.op.less_equal),
            "lt": partial(self._binary_op, op=relax.op.less),
            "iadd": partial(self._binary_op, op=relax.op.add),
            "mul": partial(self._binary_op, op=relax.op.multiply),
            "ne": partial(self._binary_op, op=relax.op.not_equal),
            "pow": partial(self._binary_op, op=relax.op.power),
            "sub": partial(self._binary_op, op=relax.op.subtract),
            "truediv": partial(self._binary_op, op=relax.op.divide),
            # baisc ops
            "astype": self.astype,
            "constant": self.constant,
            "getitem": self.getitem,
            "setitem": self.setitem,
            "shape": self.shape,
        }

    def _register_func(self, module, func: callable, m_name: str = None, f_name: str = None):
        """Register function to be traced

        Parameters
        -------
        module:
            The module of the func.
        func: callable
            The function to be traced.
        m_name: str
            The module name.
        f_name: str
            The function name.
        """

        m_name = m_name or module.__name__
        f_name = f_name or func.__name__
        t_func = register_trace_func(func, m_name, f_name)
        setattr(module, f_name, t_func)
        m_funcs = self._traced_funcs.setdefault(m_name, {"module": module, "funcs": []})
        m_funcs["funcs"].append(f_name)
        return t_func

    def _unregister_func(self, module, f_name: str, m_name: str = None):
        """Unregister function being traced

        Parameters
        -------
        module:
            The module of the func.
        f_name: str
            The function name.
        m_name: str
            The module name.
        """

        m_name = m_name or module.__name__
        trace_funcs = MSCRegistery.get(MSCRegistery.TRACE_FUNCS)
        t_name = "{}.{}".format(m_name, f_name)
        if t_name in trace_funcs:
            setattr(module, f_name, trace_funcs[t_name]["func"])
            trace_funcs.pop(t_name)
        MSCRegistery.register(MSCRegistery.TRACE_FUNCS, trace_funcs)

    def disable_trace(self):
        """Disable tracing"""

        for m_name, info in self._traced_funcs.items():
            for f_name in info["funcs"]:
                self._unregister_func(info["module"], f_name, m_name)

    def mark(self, msg: Any) -> str:
        """Mark the message with tracer info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "PARSER[{}] {}".format(self.framework(), msg)

    @property
    def convert_map(self):
        return self._convert_map

    @property
    def traced_funcs(self):
        return self._traced_funcs

    @property
    def tensor_attrs(self):
        return self._tensor_attrs

    @classmethod
    def framework(cls):
        return "base"


@msc_utils.register_trace_parser
class BasicParser(BaseParser):
    def enable_trace(self):
        """Enable tracing"""

        return None

    @classmethod
    def framework(cls):
        return "basic"
