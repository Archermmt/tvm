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
"""tvm.contrib.msc.core.utils.register"""

from typing import Any
from .namespace import MSCMap, MSCKey, MSCFramework


def register_func(name: str, func: callable, framework: str = MSCFramework.MSC):
    """Register a func for framework.

    Parameters
    ----------
    name: string
        The name for the func.
    func: callable
        The function to be registered.
    framework: string
        Should be from MSCFramework.
    """

    funcs = MSCMap.get(MSCKey.REGISTERED_FUNCS, {})
    if framework not in funcs:
        funcs[framework] = {}
    funcs[framework][name] = func
    MSCMap.set(MSCKey.REGISTERED_FUNCS, funcs)


def get_registered_func(name: str, framework: str = MSCFramework.MSC):
    """Get the registered func of framework.

    Parameters
    ----------
    name: string
        The name for the func.
    framework: string
        Should be from MSCFramework.

    Returns
    -------
    func: callable
        The registered function.
    """

    funcs = MSCMap.get(MSCKey.REGISTERED_FUNCS, {})
    if framework not in funcs:
        return None
    return funcs[framework].get(name)


def register_tool_cls(tool_cls: Any):
    """Register a tool class.

    Parameters
    ----------
    tool_cls: class
        The tool class to be registered.
    """

    tools_cls = MSCMap.get(MSCKey.REGISTERED_TOOLS_CLS, {})
    for key in ["framework", "tool_type", "tool_style"]:
        assert hasattr(tool_cls, key), "{} should be given to register tool class".format(key)
    if tool_cls.framework() not in tools_cls:
        tools_cls[tool_cls.framework()] = {}
    framework_tools = tools_cls[tool_cls.framework()]
    if tool_cls.tool_type() not in framework_tools:
        framework_tools[tool_cls.tool_type()] = {}
    tools = framework_tools[tool_cls.tool_type()]
    tools[tool_cls.tool_style()] = tool_cls
    MSCMap.set(MSCKey.REGISTERED_TOOLS_CLS, tools_cls)


def get_registered_tool_cls(framework: str, tool_type: str, tool_style: str) -> Any:
    """Get the registered tool class.

    Parameters
    ----------
    framework: string
        Should be from MSCFramework.
    tool_type: string
        The type of the tool prune| quantize| distill| debug.
    tool_style: string
        The style of the tool.

    Returns
    -------
    tool_cls: class
        The registered tool class.
    """

    tools_cls = MSCMap.get(MSCKey.REGISTERED_TOOLS_CLS, {})
    return tools_cls.get(framework, {}).get(tool_type, {}).get(tool_style)


def register_tool_method(method_cls: Any, method_style: str = "default"):
    """Register a tool method.

    Parameters
    ----------
    method_cls: class
        The method class.
    method_style: string
        The style of the method.
    """

    tools_method = MSCMap.get(MSCKey.REGISTERED_TOOLS_METHOD, {})
    assert hasattr(method_cls, "framework") and hasattr(
        method_cls, "tool_type"
    ), "framework and tool_type should be given to register tool method"
    if method_cls.framework() not in tools_method:
        tools_method[method_cls.framework()] = {}
    register_name = "{}.{}".format(method_cls.tool_type(), method_style)
    tools_method[method_cls.framework()][register_name] = method_cls
    MSCMap.set(MSCKey.REGISTERED_TOOLS_METHOD, tools_method)


def get_registered_tool_method(
    framework: str, tool_type: str, method_style: str = "default"
) -> Any:
    """Get the registered tool method.

    Parameters
    ----------
    framework: string
        Should be from MSCFramework.
    tool_type: string
        The type of the tool prune| quantize| distill| debug.
    method_style: string
        The style of the method.

    Returns
    -------
    method_cls: class
        The method class.
    """

    tools_method = MSCMap.get(MSCKey.REGISTERED_TOOLS_METHOD, {})
    register_name = "{}.{}".format(tool_type, method_style)
    return tools_method.get(framework, {}).get(register_name)
