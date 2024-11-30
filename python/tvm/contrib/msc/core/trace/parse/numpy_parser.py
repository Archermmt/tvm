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
            "numpy.sin": partial(self._unary_op, op=relax.op.sin),
            "numpy.tanh": partial(self._unary_op, op=relax.op.tanh),
            # binary ops
            "numpy.maximum": partial(self._binary_op, op=relax.op.maximum),
            "numpy.minimum": partial(self._binary_op, op=relax.op.minimum),
        }

    def enable_trace(self) -> Tuple[dict, dict]:
        """Enable tracing"""

        for f_name in dir(np):
            func = getattr(np, f_name)
            if any(k in func.__class__.__name__ for k in ["_ArrayFunctionDispatcher", "ufunc"]):
                t_func = self._register_func(np, func, f_name=f_name)
                for f_attr in dir(func):
                    if not f_attr.startswith("_"):
                        setattr(t_func, f_attr, getattr(func, f_attr))

        checker = partial(self.check_args_module, module="numpy")
        self._tensor_attrs = {
            key: {"func": getattr(np, key), "checker": checker} for key in ["max", "min"]
        }

    @classmethod
    def framework(cls):
        return "numpy"
