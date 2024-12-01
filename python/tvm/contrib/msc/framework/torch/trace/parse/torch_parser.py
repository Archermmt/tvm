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
"""tvm.contrib.msc.framework.torch.trace.torch_parser"""

from typing import Tuple, Dict, Callable
from functools import partial
from easydict import EasyDict as edict

import torch
from tvm import relax
from tvm.relax.frontend.torch.fx_translator import TorchFXImporter
from tvm.contrib.msc.core.trace import TracedNode, TracedTensor, trace_node
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core.trace import BaseParser


@msc_utils.register_trace_parser
class TorchParser(BaseParser):
    def setup(self):
        """Setup the tracer"""

        self._fx_importer = TorchFXImporter()
        return super().setup()

    def trace_reduce(self, data: TracedTensor, op: str, axis: int = 0, keepdims: bool = False):
        func = getattr(data.data, op)
        res = func(axis=axis, keepdims=keepdims)
        node = trace_node(
            "astype",
            [data],
            [{"obj": res, "module": "numpy"}],
            attrs={"axis": axis, "keepdims": keepdims},
        )
        return node.output_at(0)

    def trace_to(self, data: TracedTensor, *args, **kwargs):
        print("trace to with data " + str(data))
        print("args {} and kwargs {}".format(args, kwargs))
        raise Exception("stop here!!")
        res = data.data.astype(dtype)
        node = trace_node(
            "astype", [data], [{"obj": res, "module": "numpy"}], attrs={"dtype": dtype}
        )
        return node.output_at(0)

    def _wrapped_converter(self, node: TracedNode, converter: callable) -> relax.Var:
        """Wrap the converter and convert node

        Parameters
        -------
        node: TracedNode
            The traced node.
        converter: func
            The converter of TorchFXImporter.

        Returns
        -------
        expr: relax.Expr
            The converted expr.
        """

        inputs = self.retrieve_args(node)
        for expr, tensor in zip(inputs, node.get_inputs()):
            if tensor.name not in self._fx_importer.env:
                self._fx_importer.env[tensor.name] = expr
        self._fx_importer.block_builder = self._tracer.block_builder
        fx_node = edict(args=[i.name for i in node.get_inputs()], kwargs=node.attrs)
        res = converter(fx_node)
        if node.attrs.get("multi_outputs", False):
            for idx, o in enumerate(node.get_outputs()):
                self._fx_importer.env[o.name] = res[idx]
        else:
            self._fx_importer.env[node.output_at(0).name] = res
        return res

    def _create_convert_map(self) -> Dict[str, Callable[[TracedNode], relax.Var]]:
        """Create convert map for nodes

        Returns
        -------
        convert_map: list<str, callable>
            The convert_map.
        """

        convert_map = {}
        for name, converter in self._fx_importer.convert_map.items():
            if not isinstance(name, str):
                continue
            if hasattr(torch, name):
                convert_map["torch." + name] = partial(self._wrapped_converter, converter=converter)
            elif hasattr(torch.nn.functional, name):
                convert_map["torch.nn.functional." + name] = partial(
                    self._wrapped_converter, converter=converter
                )
        return convert_map

    def enable_trace(self) -> Tuple[dict, dict]:
        """Enable tracing"""

        recorded_funcs = set()

        def _check_func(f_name, func):
            if not callable(func):
                return False
            if f_name.startswith("__"):
                return False
            if func.__class__.__name__ not in ("function", "builtin_function_or_method"):
                return False
            if func in recorded_funcs:
                return False
            recorded_funcs.add(func)
            return True

        for f_name in dir(torch):
            func = getattr(torch, f_name)
            if not _check_func(f_name, func):
                continue
            self._register_func(torch, func, f_name=f_name)
        for f_name in dir(torch.nn.functional):
            func = getattr(torch.nn.functional, f_name)
            if not _check_func(f_name, func):
                continue
            self._register_func(torch.nn.functional, func, f_name=f_name)

        checker = partial(self.check_args_module, module="torch")
        t_attrs = ["max", "mean", "min", "sum"]
        self._tensor_attrs = {
            key: {"func": partial(self.trace_reduce, op=key), "checker": checker} for key in t_attrs
        }
        self._tensor_attrs["to"] = {"func": self.trace_to, "checker": checker}

    @classmethod
    def framework(cls):
        return "torch"
