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

from functools import partial
from typing import Dict, Callable
import numpy as np

from tvm import relax
from tvm.contrib.msc.core import utils as msc_utils
from ..graph import *
from .base_parser import BaseParser


@msc_utils.register_trace_parser
class NumpyParser(BaseParser):
    def trace_astype(self, data: TracedTensor, dtype: str):
        res = data.data.astype(dtype)
        node = trace_node(
            "astype", [data], [{"obj": res, "module": "numpy"}], attrs={"dtype": dtype}
        )
        return node.output_at(0)

    def _reduce_op(self, node: TracedNode, op: Callable) -> relax.Var:
        axis = node.get_attr("axis")
        if axis is None and node.get_attr:
            axis = node.get_arg_attr(1)
        keepdims = node.get_attr("keepdims")
        if keepdims is None:
            keepdims = node.get_arg_attr(2, False)
        data = self.retrieve_arg(node.input_at(0))
        return self.emit(op(data, axis=axis, keepdims=keepdims))

    def argsort(self, node: TracedNode) -> relax.Var:
        data = self.retrieve_arg(node.input_at(0))
        axis = node.get_attr("axis")
        if axis is None:
            axis = node.get_arg_attr(1)
        return relax.op.argsort(data, axis, dtype=node.output_at(0).dtype)

    def full(self, node: TracedNode) -> relax.Var:
        shape = self.retrieve_arg(node.input_at(0))
        fill_value = node.get_attr("fill_value") or node.get_arg_attr(1)
        dtype = node.get_attr("dtype") or node.get_arg_attr(2)
        fill_value = relax.const(fill_value, str(dtype))
        return relax.op.full(shape, fill_value, str(dtype))

    def full_like(self, node: TracedNode) -> relax.Var:
        data = self.retrieve_arg(node.input_at(0))
        dtype = node.input_at(0).dtype
        fill_value = node.get_attr("fill_value") or node.get_arg_attr(1)
        fill_value = relax.const(fill_value, dtype)
        return relax.op.full_like(data, fill_value, dtype=dtype)

    def ones(self, node: TracedNode) -> relax.Var:
        shape = self.retrieve_arg(node.input_at(0))
        dtype = node.get_attr("dtype") or node.get_arg_attr(1)
        return relax.op.ones(shape, str(dtype))

    def ones_like(self, node: TracedNode) -> relax.Var:
        data = self.retrieve_arg(node.input_at(0))
        dtype = node.get_attr("dtype") or node.input_at(0).dtype
        return relax.op.ones_like(data, dtype)

    def sort(self, node: TracedNode) -> relax.Var:
        data = self.retrieve_arg(node.input_at(0))
        axis = node.get_attr("axis")
        if axis is None:
            axis = node.get_arg_attr(1)
        return relax.op.sort(data, axis)

    def zeros(self, node: TracedNode) -> relax.Var:
        shape = self.retrieve_arg(node.input_at(0))
        dtype = node.get_attr("dtype") or node.get_arg_attr(1)
        return relax.op.zeros(shape, str(dtype))

    def zeros_like(self, node: TracedNode) -> relax.Var:
        data = self.retrieve_arg(node.input_at(0))
        dtype = node.get_attr("dtype") or node.input_at(0).dtype
        return relax.op.zeros_like(data, dtype)

    def _create_convert_map(self) -> Dict[str, Callable[[TracedNode], relax.Var]]:
        """Create convert map for nodes

        Returns
        -------
        convert_map: list<str, callable>
            The convert_map.
        """

        return {
            # unary ops
            "numpy.abs": partial(self._unary_op, op=relax.op.abs),
            "numpy.arccos": partial(self._unary_op, op=relax.op.acos),
            "numpy.arccosh": partial(self._unary_op, op=relax.op.acosh),
            "numpy.arcsin": partial(self._unary_op, op=relax.op.asin),
            "numpy.arcsinh": partial(self._unary_op, op=relax.op.asinh),
            "numpy.arctan": partial(self._unary_op, op=relax.op.atan),
            "numpy.arctanh": partial(self._unary_op, op=relax.op.atanh),
            "numpy.cos": partial(self._unary_op, op=relax.op.cos),
            "numpy.exp": partial(self._unary_op, op=relax.op.exp),
            "numpy.round": partial(self._unary_op, op=relax.op.round),
            "numpy.sin": partial(self._unary_op, op=relax.op.sin),
            "numpy.tanh": partial(self._unary_op, op=relax.op.tanh),
            # binary ops
            "numpy.maximum": partial(self._binary_op, op=relax.op.maximum),
            "numpy.minimum": partial(self._binary_op, op=relax.op.minimum),
            # reduce ops
            "numpy.argmax": partial(self._reduce_op, op=relax.op.argmax),
            "numpy.argmin": partial(self._reduce_op, op=relax.op.argmin),
            "numpy.max": partial(self._reduce_op, op=relax.op.max),
            "numpy.mean": partial(self._reduce_op, op=relax.op.mean),
            "numpy.min": partial(self._reduce_op, op=relax.op.min),
            "numpy.sum": partial(self._reduce_op, op=relax.op.sum),
            # numpy ops
            "numpy.argsort": self.argsort,
            "numpy.full": self.full,
            "numpy.full_like": self.full_like,
            "numpy.ones": self.ones,
            "numpy.ones_like": self.ones_like,
            "numpy.sort": self.sort,
            "numpy.zeros": self.zeros,
            "numpy.zeros_like": self.zeros_like,
        }

    def enable_trace(self) -> Tuple[dict, dict]:
        """Enable tracing"""

        def _check_func(f_name: str, func: callable):
            if any(k in func.__class__.__name__ for k in ["_ArrayFunctionDispatcher", "ufunc"]):
                return True
            if f_name in ["ones", "zeros", "full"]:
                return True
            return False

        for f_name in dir(np):
            func = getattr(np, f_name)
            if not _check_func(f_name, func):
                continue
            t_func = self._register_func(np, func, f_name=f_name)
            for f_attr in dir(func):
                if not f_attr.startswith("_"):
                    setattr(t_func, f_attr, getattr(func, f_attr))

        checker = partial(self.check_args_module, module="numpy")
        t_attrs = ["max", "mean", "min", "sum", "argmax", "argmin"]
        self._tensor_attrs = {
            key: {"func": getattr(np, key), "checker": checker} for key in t_attrs
        }
        self._tensor_attrs["astype"] = {"func": self.trace_astype, "checker": checker}

    @classmethod
    def framework(cls):
        return "numpy"
