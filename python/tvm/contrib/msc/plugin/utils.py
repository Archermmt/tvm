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
"""tvm.contrib.msc.plugin.utils"""

from typing import Any

from tvm import relax
from tvm import tir


def to_expr(value: Any) -> relax.Expr:
    """Change value to expr

    Parameters
    ----------
    value:
        The value with python type.

    Returns
    -------
    expr: relax.Expr
        The relax Expr.
    """

    if isinstance(value, (bool, int)):
        value = tir.IntImm("int64", value)
        expr = relax.PrimValue(value)
    elif isinstance(value, float):
        value = tir.FloatImm("float64", value)
        expr = relax.PrimValue(value)
    elif isinstance(value, str):
        expr = relax.StringImm(value)
    elif isinstance(value, (list, tuple)):
        expr = relax.Tuple([to_expr(v) for v in value])
    else:
        raise TypeError(f"Unsupported input type: {type(value)}")
    return expr
