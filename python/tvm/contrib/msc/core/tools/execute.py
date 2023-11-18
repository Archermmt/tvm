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
"""tvm.contrib.msc.core.tools.execute"""

from functools import wraps
from typing import List, Iterable, Any, Dict

import tvm
from tvm.contrib.msc.core.utils.namespace import MSCMap, MSCKey
from tvm.contrib.msc.core import utils as msc_utils
from .tool import ToolType, BaseTool


def _get_tool_key(tool_type: str) -> str:
    """Get the key according to tool_type

    Parameters
    -------
    tool_type: str
        The type of the tool prune| quantize| distill...

    Returns
    -------
    tool_key: str
        The tool key.
    """

    if tool_type == ToolType.PRUNE:
        return MSCKey.PRUNER
    if tool_type == ToolType.QUANTIZE:
        return MSCKey.QUANTIZER
    if tool_type == ToolType.DISTILL:
        return MSCKey.DISTILLER
    if tool_type == ToolType.DEBUG:
        return MSCKey.DEBUGGER
    raise TypeError("Unexpected tool type " + str(tool_type))


def add_tool(tool: BaseTool, tool_type: str, tag: str = "main"):
    """Add tool by type and tag

    Parameters
    -------
    tool: BaseTool
        The tool.
    tool_type: str
        The type of the tool prune| quantize| distill...
    tag: str
        The tag of the tool.
    """

    tool_key = _get_tool_key(tool_type)
    tools = MSCMap.get(tool_key, {})
    tools[tag] = tool
    MSCMap.set(tool_key, tools)
    return tool


def create_tool(framework: str, tool_type: str, config: dict, tag: str = "main") -> BaseTool:
    """Create tool by type, config and tag

    Parameters
    -------
    framework: str
        The framework for implement
    tool_type: str
        The type of the tool prune| quantize| distill...
    config: dict
        The config of tool.
    tag: str
        The tag of the tool.
    """

    tool_style = config.pop("tool_style") if "tool_style" in config else "default"
    tool_cls = msc_utils.get_registered_tool_cls(framework, tool_type, tool_style)
    assert tool_cls, "Can not find tool class for {}:{}".format(tool_type, tool_style)
    return add_tool(tool_cls(**config), tool_type, tag)


def get_tool(tool_type: str, tag: str = "main") -> BaseTool:
    """Get tool by type and tag

    Parameters
    -------
    tool_type: str
        The type of the tool prune| quantize| distill...
    tag: str
        The tag of the tool.

    Returns
    -------
    tool: BaseTool
        The saved tool.
    """

    tool_key = _get_tool_key(tool_type)
    tools = MSCMap.get(tool_key, {})
    return tools.get(tag)


def get_tools(tag: str = "main") -> Iterable[BaseTool]:
    """Get all saved tools by tag

    Parameters
    -------
    tag: str
        The tag of the tool.

    Returns
    -------
    tools: iterable<BaseTool>
        The saved tools.
    """

    for t_type in ToolType.all_types():
        tool = get_tool(t_type, tag)
        if tool:
            yield tool


def execute_tool(stage: str, tag: str = "main") -> callable:
    """Wrapper for tool execution

    Parameters
    -------
    stage: str
        The stage for tool execution build| forward
    tag: str
        The tag of the tool.

    Returns
    -------
    decorate: callable
        The decorate.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for tool in get_tools(tag):
                if stage == "build":
                    tool.execute_before_build(*args, **kwargs)
                elif stage == "forward":
                    tool.execute_before_forward(*args, **kwargs)
                else:
                    raise TypeError("Unexpected stage " + str(stage))
            output = func(*args, **kwargs)
            for tool in get_tools(tag):
                if stage == "build":
                    output = tool.execute_after_build(output)
                elif stage == "forward":
                    output = tool.execute_after_forward(output)
                else:
                    raise TypeError("Unexpected stage " + str(stage))
            return output

        return wrapper

    return decorate


def process_tensor(tensor: Any, name: str, consumer: str, phase: str, tag: str = "main") -> Any:
    """Process tensor with tools

    Parameters
    -------
    tensor: Any
        Tensor in framework
    name: str
        The name of the tensor.
    consumer: str
        The name of the consumer.
    phase: str
        The phase mark teacher| student| null
    tag: str
        The tag of the tool.

    Returns
    -------
    tensor: Any
        The processed tensor.
    """

    for tool in get_tools(tag):
        tensor = tool.process_tensor(tensor, name, consumer, phase)
    return tensor


@tvm.register_func("msc_tool.execute_hook")
def execute_hook(datas: Dict[str, tvm.nd.array], graph_name: str, stage: str, tag: str = "main"):
    """Hook for tool execution

    Parameters
    -------
    datas: dict<str, tvm.nd.array>
        The datas to be processed
    graph_name: str
        The graph name.
    stage: str
        The stage for tool execution build| forward
    tag: str
        The tag of the tool.
    """

    for tool in get_tools(tag):
        if stage == "before_build":
            tool.execute_before_build(datas, graph_name=graph_name)
        elif stage == "after_build":
            tool.execute_after_build(datas, graph_name=graph_name)
        elif stage == "before_forward":
            tool.execute_before_forward(datas, graph_name=graph_name)
        elif stage == "after_forward":
            tool.execute_after_forward(datas, graph_name=graph_name)
        else:
            raise TypeError("Unexpected stage " + str(stage))


@tvm.register_func("msc_tool.process_tensor_codegen")
def process_tensor_codegen(
    tensor_ctx: Dict[str, str], name: str, consumer: str, phase: str, tag: str = "main"
) -> List[str]:
    """Codegen processed tensor describe with tools

    Parameters
    -------
    tensor_ctx: dict<str, str>
        Tensor describe items.
    name: str
        The name of the tensor.
    consumer: str
        The name of the consumer.
    phase: str
        The phase mark teacher| student| null
    tag: str
        The tag of the tool.

    Returns
    -------
    processed: list<str>
        The tensor describe for processed tensor.
    """

    tensor = process_tensor(tensor_ctx, name, consumer, phase, tag)
    return tensor.get("processed", [])
