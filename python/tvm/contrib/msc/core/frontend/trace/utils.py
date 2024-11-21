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
"""tvm.contrib.msc.core.frontend.trace.utils"""

from typing import List, Dict, Any
from tvm.contrib.msc.core.utils.namespace import MSCMap, MSCKey


def set_global_tracer(tracer: "Tracer") -> "Tracer":
    """Set the global tracer

    Parameters
    ----------
    tracer: Tracer
        The tracer to be set.

    Returns
    -------
    tracer: Tracer
        The setted tracer.
    """

    MSCMap.set(MSCKey.GLOBAL_TRACER, tracer)
    return tracer


def get_global_tracer() -> "Tracer":
    """Get the tracer

    Returns
    -------
    tracer: Tracer
        The tracer.
    """

    return MSCMap.get(MSCKey.GLOBAL_TRACER)


def trace_node(
    optype: str,
    inputs: List[Any],
    outputs: List[Any],
    name: str = None,
    attrs: Dict[str, Any] = None,
    meta: Any = None,
):
    """Add node in tracer graph

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

    tracer = get_global_tracer()
    assert tracer, "Missing tracer for add node"
    return tracer.graph.add_node(optype, inputs, outputs, name, attrs, meta)
