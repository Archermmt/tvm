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
# pylint: disable=unused-argument
"""tvm.contrib.msc.framework.tensorrt.transform.pattern"""

from typing import Mapping, Tuple, List, Union, Callable

from tvm import relax
from tvm.relax.dpl import pattern
from tvm.relax.transform import PatternCheckContext, FusionPattern
from tvm.relax.backend.pattern_registry import register_patterns


def basic_pattern(
    op_name: str, input_types: List[str] = None
) -> Tuple[pattern.DFPattern, Mapping[str, pattern.DFPattern]]:
    """create basic pattern for tensorrt support ops.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.nn.conv2d"
    input_types: list<str>
        The input types, elach element can be input| constant

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a conv_bias operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    input_types = input_types or ["input"]
    inputs = []
    for i_type in input_types:
        if i_type == "input":
            inputs.append(pattern.wildcard())
        elif i_type == "constant":
            inputs.append(pattern.is_const())
        else:
            raise Exception("Unexpected input type " + str(i_type))
    out = pattern.is_op(op_name)(*inputs)
    annotations = {"input_" + str(idx): arg for idx, arg in enumerate(inputs)}
    annotations["out"] = out
    return out, annotations


def conv2d_bias_pattern() -> Tuple[pattern.DFPattern, Mapping[str, pattern.DFPattern]]:
    """Create patterns for an conv2d fused with bias.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a conv_bias operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    data = pattern.wildcard()
    weight = pattern.is_const()
    conv = pattern.is_op("relax.nn.conv2d")(data, weight)
    bias = pattern.is_const()
    out = pattern.is_op("relax.add")(conv, bias)
    annotations = {"data": data, "weight": weight, "bias": bias, "conv": conv, "out": out}
    return out, annotations


def _basic_check(context: PatternCheckContext) -> bool:
    """Check if the basic pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    for _, expr in context.annotated_expr.items():
        if isinstance(expr, relax.ShapeExpr):
            continue
        if any(i < 0 for i in expr.struct_info.shape.values):
            return False
        if expr.struct_info.dtype not in ("float32", "float16"):
            return False
    return True


def _elemwise_check(context: PatternCheckContext) -> bool:
    """Check if the elemwise pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    if not _basic_check(context):
        return False
    ndim_a = len(context.annotated_expr["input_0"].struct_info.shape.values)
    ndim_b = len(context.annotated_expr["input_1"].struct_info.shape.values)
    return ndim_a == ndim_b


def _conv2d_bias_check(context: PatternCheckContext) -> bool:
    """Check if the conv2d_bias pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    if not _basic_check(context):
        return False
    ndim_conv = len(context.annotated_expr["conv"].struct_info.shape.values)
    ndim_bias = len(context.annotated_expr["bias"].struct_info.shape.values)
    ndim_out = len(context.annotated_expr["out"].struct_info.shape.values)
    return ndim_conv == ndim_bias and ndim_bias == ndim_out


CheckFunc = Callable[[Mapping[pattern.DFPattern, relax.Expr], relax.Expr], bool]
Pattern = Union[
    FusionPattern,
    Tuple[str, pattern.DFPattern],
    Tuple[str, pattern.DFPattern, Mapping[str, pattern.DFPattern]],
    Tuple[str, pattern.DFPattern, Mapping[str, pattern.DFPattern], CheckFunc],
]


def get_patterns(target) -> List[Pattern]:
    """Get all the tensorrt patterns.

    Parameters
    ----------
    target: str
        The target name for tensorrt patterns.

    Returns
    -------
    patterns: list<Pattern>
        The patterns
    """

    basic_ops = {"nn.conv2d": ["input", "constant"], "reshape": ["input", "input"]}
    elemwise_ops = {
        "add": ["input", "input"],
        "divide": ["input", "input"],
        "multiply": ["input", "input"],
        "subtract": ["input", "input"],
    }
    patterns = []
    # basic ops
    for op, in_types in basic_ops.items():
        patterns.append((target + "." + op, *basic_pattern("relax." + op, in_types), _basic_check))
    # elemwise ops
    for op, in_types in elemwise_ops.items():
        patterns.append(
            (target + "." + op, *basic_pattern("relax." + op, in_types), _elemwise_check)
        )
    # fusable ops
    patterns.extend([(target + ".msc.conv2d_bias", *conv2d_bias_pattern(), _conv2d_bias_check)])
    return patterns


register_patterns(get_patterns("msc_tensorrt"))
