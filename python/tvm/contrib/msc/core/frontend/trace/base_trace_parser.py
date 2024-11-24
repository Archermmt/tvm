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
"""tvm.contrib.msc.core.frontend.trace.base_trace_parser"""

import logging
from functools import partial
from typing import Any, Dict, Callable

from tvm import relax
from tvm.contrib.msc.core import utils as msc_utils
from .graph import *


class BaseTraceParser(object):
    """Parser for tracing

    Parameters
    ----------
    tracer: str
        The tracer.
    verbose: str
        The verbose level.
    logger: logging.Logger
        The logger.
    """

    def __init__(
        self, tracer: "Tracer", config, verbose: str = "info", logger: logging.Logger = None
    ):
        self._tracer = tracer
        self._config = config
        self._verbose = verbose
        self._logger = logger or msc_utils.get_global_logger()
        self._logger.info(msc_utils.msg_block(self.mark("SETUP"), self.setup()))

    def setup(self):
        """Setup the tracer"""

        self._convert_map = self._create_convert_map()
        return {"converters": len(self._convert_map)}

    def convert(self, node: TracedNode) -> relax.Expr:
        """Convert node to relax Expr"""

        assert node.optype in self._convert_map, "Convert is not support for {}".format(node.optype)
        expr = self._convert_map[node.optype](node)
        if isinstance(expr, (relax.Var, relax.Constant)):
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
            The name hint of expr

        Returns
        -------
        var: relax.Var
            The emitted var..
        """

        if name_hint:
            return self._tracer.block_builder.emit(expr, name_hint)
        return self._tracer.block_builder.emit(expr)

    def _binary_op(self, node: TracedNode, op: Callable) -> relax.Var:
        inputs = self.retrieve_args(node)
        assert len(inputs) == 2, "binary op only support 2 inputs, get " + str(inputs)
        return self.emit(op(*inputs))

    def _constant(self, node: TracedNode) -> relax.Var:
        scalar, output = node.get_attr("scalar"), node.output_at(0)
        if scalar is None:
            return relax.const(msc_utils.cast_array(output.data), output.dtype)
        return relax.const(scalar, output.dtype)

    def _create_convert_map(self) -> Dict[str, Callable[[TracedNode], relax.Var]]:
        """Create convert map for nodes

        Returns
        -------
        convert_map: list<str, callable>
            The convert_map.
        """

        return {
            "constant": self._constant,
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
            "pow": partial(self._binary_op, op=relax.op.power),
            "sub": partial(self._binary_op, op=relax.op.subtract),
            "truediv": partial(self._binary_op, op=relax.op.divide),
        }

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

    @classmethod
    def framework(cls):
        return "base"
