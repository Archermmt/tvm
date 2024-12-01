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
from functools import partial

from tvm.contrib.msc.core.utils.namespace import MSCMap, MSCKey
from tvm.contrib.msc.core.utils.register import MSCRegistery


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


def traced_func(*args, trace_name: str = None, **kwargs):
    """The traced function

    Parameters
    ----------
    name: str
        The name of the func.
    """

    info = MSCRegistery.get(MSCRegistery.TRACE_FUNCS, {}).get(trace_name)
    assert info, "Missing {} in trace funcs".format(trace_name)
    inputs, attrs = [], {}
    raw_args, raw_kwargs = [], {}
    for arg in args:
        if arg.__class__.__name__ == "TracedTensor":
            raw_args.append(arg.data)
            inputs.append(arg)
            attrs.setdefault("args", []).append(arg.name)
        else:
            raw_args.append(arg)
            attrs.setdefault("args", []).append(arg)
    for k, v in kwargs.items():
        if v.__class__.__name__ == "TracedTensor":
            raw_kwargs[k] = v.data
            inputs.append(v)
        else:
            raw_kwargs[k] = v
            attrs[k] = v
    results = info["func"](*raw_args, **raw_kwargs)
    if isinstance(results, (tuple, list)):
        attrs["multi_outputs"] = True
    else:
        results = [results]
    results = [{"obj": r, "module": info["module"]} for r in results]
    node = trace_node(trace_name, inputs, results, attrs=attrs)
    outputs = node.get_outputs()
    if attrs.get("multi_outputs", False):
        return outputs
    return outputs[0]


def register_trace_func(func: Any, m_name: str, f_name: str):
    """Register a func for tracing

    Parameters
    ----------
    func: callable
        The func to be register.
    m_name: str
        The name of module.
    f_name: str
        The name of function.
    """

    t_name = "{}.{}".format(m_name, f_name)
    trace_funcs = MSCRegistery.get(MSCRegistery.TRACE_FUNCS, {})
    if t_name in trace_funcs:
        return func
    trace_funcs[t_name] = {"func": func, "module": m_name}
    MSCRegistery.register(MSCRegistery.TRACE_FUNCS, trace_funcs)
    return partial(traced_func, trace_name=t_name)
