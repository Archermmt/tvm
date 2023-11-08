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
"""tvm.contrib.msc.core.tools.base_tool"""

import os
import copy
import logging
from functools import wraps
from typing import List, Iterable, Any, Tuple, Dict
import numpy as np

import tvm
from tvm.contrib.msc.core.ir import MSCGraph, MSCJoint, MSCTensor
from tvm.contrib.msc.core.utils.namespace import MSCMap, MSCKey, MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class MSCToolType(object):
    """Enum all msc tool typs"""

    BASE = "base"
    PRUNE = "prune"
    QUANTIZE = "quantize"
    DISTILL = "distill"
    DEBUG = "debug"

    @classmethod
    def all_types(cls) -> List[str]:
        return [cls.PRUNE, cls.QUANTIZE, cls.DISTILL, cls.DEBUG]


class MSCToolImpl(object):
    def setup(self, options: dict):
        """Setup the implement"""

        self._options = options or {}

    def reset(self):
        """reset the implement"""

        return

    def execute_before_build(self, *args, **kwargs):
        """Execute before model build

        Parameters
        ----------
        args: list<Any>
            The arguments for model build.
        kwargs: dict<Any>
            The key word arguments for model build.
        """

        return

    def execute_after_build(self, output: Any) -> Any:
        """Execute after model build

        Parameters
        ----------
        output: Any
            The output reference of the model.

        Returns
        -------
        output: Any
           The modified output reference.
        """

        return output

    def execute_before_forward(self, *args, **kwargs):
        """Execute before model forward

        Parameters
        ----------
        args: list<Any>
            The arguments for model forward.
        kwargs: dict<Any>
            The key word arguments for model forward.
        """

        return

    def execute_after_forward(self, output: Any) -> Any:
        """Execute after model forward

        Parameters
        ----------
        output: Any
            The output reference of the model.

        Returns
        -------
        output: Any
           The modified output reference.
        """

        return output

    @classmethod
    def tool_type(cls):
        return MSCToolType.BASE

    @classmethod
    def framework(cls):
        return MSCFramework.MSC


class MSCTool(object):
    """Basic tool of MSC

    Parameters
    ----------
    tool_impl: MSCToolImpl
        The implement of the tool
    plan_file: str
        The plan file path.
    strategy: dict
        The strategy of the tool.
    workspace: MSCDirectory
        The workspace of the tool.
    options: dict
        The extra options for the tool
    verbose_step: int
        The verbose interval step.
    logger: logging.Logger
        The logger
    """

    def __init__(
        self,
        tool_impl: MSCToolImpl,
        plan_file: str,
        strategy: dict,
        workspace: msc_utils.MSCDirectory,
        options: dict = None,
        verbose_step: int = 50,
        logger: logging.Logger = None,
    ):
        self._tool_impl = tool_impl
        if os.path.isfile(plan_file):
            self._plan = msc_utils.load_dict(plan_file)
        else:
            self._plan = {}
        self._methods, self._method_configs = self._parse_strategy(msc_utils.copy_dict(strategy))
        self._workspace = workspace
        self._verbose_step = verbose_step
        self._logger = logger or msc_utils.get_global_logger()
        self.setup(options)
        init_title = "{}.INIT ({})".format(self.tool_type().upper(), self._tool_impl.framework())
        init_info = {
            "style": self.tool_style(),
            "methods": self._methods,
            "method_configs": self._method_configs,
            "options": self._options,
            "verbose_step": self._verbose_step,
            "planed_num": len(self._plan),
        }
        self._logger.info(msc_utils.msg_block(init_title, init_info))
        if self._plan:
            self._logger.debug(
                msc_utils.msg_block("{}.PLAN".format(self.tool_type().upper()), self._plan)
            )

    def setup(self, options: dict):
        """Setup the tool

        Parameters
        ----------
        options: dict
            The options for setup the tool
        """

        self._options = options or {}
        self._tensor_status = {}
        self._enabled, self._is_training = True, True
        self._graphs, self._weights = [], []
        self._graph_id, self._forward_cnt = 0, 0
        self._tool_impl.setup(options)

    def reset(
        self, graphs: List[MSCGraph], weights: List[Dict[str, tvm.nd.array]]
    ) -> Tuple[List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
        """Reset the tool

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        """

        self._forward_cnt = 0
        self._graphs = graphs
        self._weights = weights
        return self._graphs, self._weights

    def execute_before_build(self, *args, **kwargs):
        """Execute before model build

        Parameters
        ----------
        args: list<Any>
            The arguments for model build.
        kwargs: dict<Any>
            The key word arguments for model build.
        """

        if self._enabled:
            self._graph_id = self._infer_graph_id(kwargs)
            self._logger.debug(
                "<{}> before build graph[{}]({})".format(
                    self.tool_type(), self._graph_id, "train" if self._is_training else "eval"
                )
            )
            self._tool_impl.execute_before_build(*args, **kwargs)

    def execute_after_build(self, output: Any) -> Any:
        """Execute after model build

        Parameters
        ----------
        output: Any
            The output reference of the model.

        Returns
        -------
        output: Any
           The modified output reference.
        """

        if self._enabled:
            output = self._tool_impl.execute_after_build(output)
            self._logger.debug(
                "<{}> after build graph[{}]({})".format(
                    self.tool_type(), self._graph_id, "train" if self._is_training else "eval"
                )
            )
        return output

    def execute_before_forward(self, *args, **kwargs):
        """Execute before model forward

        Parameters
        ----------
        args: list<Any>
            The arguments for model forward.
        kwargs: dict<Any>
            The key word arguments for model forward.
        """

        if self._enabled:
            self._graph_id = self._infer_graph_id(kwargs)
            if self.should_log():
                self._logger.debug(
                    "<{}> start graph[{}] forward[{}]".format(
                        self.tool_type(), self._graph_id, self._forward_cnt
                    )
                )
            self._tool_impl.execute_before_forward(*args, **kwargs)

    def execute_after_forward(self, output: Any) -> Any:
        """Execute after model forward

        Parameters
        ----------
        output: Any
            The output reference of the model.

        Returns
        -------
        output: Any
           The modified output reference.
        """

        if self._enabled:
            output = self._tool_impl.execute_after_forward(output)
            if self.should_log():
                self._logger.debug(
                    "<{}> end graph[{}] forward[{}]".format(
                        self.tool_type(), self._graph_id, self._forward_cnt
                    )
                )
            self._forward_cnt += 1
        return output

    def process_tensor(self, tensor: Any, name: str, consumer: str, phase: str) -> Any:
        """Process tensor

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

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        edge_id = self.to_edge_id(name, consumer)
        if edge_id not in self._tensor_status:
            self._tensor_status[edge_id] = {}
        if "process" not in self._tensor_status[edge_id]:
            self._tensor_status[edge_id]["process"] = self._check_tensor(name, consumer, phase)
            self._logger.debug(
                "Update tensor status(process) {}: {}".format(edge_id, self._tensor_status[edge_id])
            )
        if not self._tensor_status[edge_id]["process"]:
            return tensor
        return self._tool_impl.process_tensor(tensor, name, consumer, phase)

    def visualize(self, visual_dir: msc_utils.MSCDirectory):
        """Visualize MSCGraphs

        Parameters
        -------
        visual_dir: MSCDirectory
            Visualize path for saving graph
        """

        return

    def _check_tensor(self, name: str, consumer: str, phase: str) -> bool:
        """Check if the tensor should be processed

        Parameters
        -------
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        phase: str
            The phase mark teacher| student| null

        Returns
        -------
        vaild: bool
            Whether to process the tensor.
        """

        return True

    def update_plan(self, plan: dict):
        """Update the plan

        Parameters
        ----------
        plan: dict
            The new plan.
        """

        self._plan.update(plan)

    def create_plan(self) -> dict:
        """Create the plan

        Returns
        -------
        plan: dict
            THe plan of the tool.
        """

        return self._plan

    def enable(self):
        """Enable the tool"""

        self._enabled = True

    def disable(self):
        """Disable the tool"""

        self._enabled = False

    def train(self):
        """Set the tool to train mode"""

        self._is_training = True

    def eval(self):
        """Set the tool to eval mode"""

        self._is_training = False

    def to_edge_id(self, name: str, consumer: str) -> str:
        """Concat name to unique id

        Parameters
        ----------
        name: str
            The name of tensor.
        consumer: str
            The name of consumer.

        Returns
        -------
        edge_id: str
           The unique name of edge.
        """

        return "{}-c-{}".format(name, consumer)

    def from_edge_id(self, edge_id: str) -> Tuple[str]:
        """Concat name to unique id

        Parameters
        ----------
        edge_id: str
           The unique name of edge.

        Returns
        -------
        name: str
            The name of tensor.
        consumer: str
            The name of consumer.
        """

        return edge_id.split("-c-")

    def should_log(self) -> bool:
        """Check if should log

        Returns
        -------
        should_log: bool
           Whether should log.
        """

        return self._forward_cnt % self._verbose_step == 0

    def _infer_graph_id(self, kwargs: dict) -> int:
        """Infer graph id from kwargs

        Parameters
        ----------
        kwargs: dict
           The kwargs for execute.
        """

        if "graph_name" in kwargs:
            for idx, g in enumerate(self._graphs):
                if g.name == kwargs["graph_name"]:
                    return idx
        return kwargs.get("graph_id", 0)

    def get_nodes(self) -> Iterable[MSCJoint]:
        """Get all the nodes in the graphs.

        Returns
        -------
        nodes: generator<MSCJoint>
            The generator of nodes.
        """

        for g in self._graphs:
            for n in g.get_nodes():
                yield n

    def find_node(self, name: str) -> MSCJoint:
        """Find node by name.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        node: MSCJoint
            The found node.
        """

        for g in self._graphs:
            if g.has_node(name):
                return g.find_node(name)
        raise Exception("Can not find node {} from graphs".format(name))

    def find_tensor(self, name: str) -> MSCTensor:
        """Find tensor by name.

        Parameters
        ----------
        name: string
            The name of the tensor.

        Returns
        -------
        node: MSCTensor
            The found tensor.
        """

        for g in self._graphs:
            if g.has_tensor(name):
                return g.find_tensor(name)
        raise Exception("Can not find tensor {} from graphs".format(name))

    def find_producer(self, name: str) -> MSCJoint:
        """Find producer by tensor_name .

        Parameters
        ----------
        name: string
            The name of the tensor.

        Returns
        -------
        node: MSCJoint
            The found prducer.
        """

        for g in self._graphs:
            if g.has_tensor(name):
                return g.find_producer(name)
        raise Exception("Can not find producer of {} from graphs".format(name))

    def find_consumers(self, name: str) -> List[MSCJoint]:
        """Find consumers by tensor_name.

        Parameters
        ----------
        name: string
            The name of the tensor.

        Returns
        -------
        node: list<MSCJoint>
            The found consumers.
        """

        for g in self._graphs:
            if g.has_tensor(name):
                return g.find_consumers(name)
        raise Exception("Can not find consumers of {} from graphs".format(name))

    def get_data(self, name: str) -> np.ndarray:
        """Get the data by name

        Parameters
        -------
        name: str
            The tensor name

        Returns
        -------
        data: np.ndarray
            The data.
        """

        for sub_weights in self._weights:
            if name in sub_weights:
                return msc_utils.cast_array(sub_weights[name])
        raise Exception("Can not find data " + str(name))

    def _parse_strategy(self, strategy: dict):
        """Parse the strategy to get valid strategy

        Parameters
        -------
        strategy: dict
            The given strategy

        Returns
        -------
        method_configs: dict
            The parsed method configs.
        methods: dict
            The parsed methods.
        """

        methods, method_configs = {}, {}
        assert isinstance(strategy, list) and all(
            isinstance(s, dict) for s in strategy
        ), "Strategy should be given as list of dict"
        for s in strategy:
            method_cls_name = s.pop("method_cls") if "method_cls" in s else "default"
            method_cls = msc_utils.get_registered_tool_method(
                self._tool_impl.framework(), self.tool_type(), method_cls_name
            )
            default_cls = msc_utils.get_registered_tool_method(
                MSCFramework.MSC, self.tool_type(), method_cls_name
            )
            method = s.pop("method") if "method" in s else "default"
            method = (
                getattr(method_cls, method)
                if hasattr(method_cls, method)
                else getattr(default_cls, method)
            )
            if "types" in s:
                types = s.pop("types")
                methods.update({s_type: method for s_type in types})
                method_configs.update({s_type: copy.deepcopy(s) for s_type in types})
            elif "names" in s:
                names = s.pop("names")
                methods.update({s_name: method for s_name in names})
                method_configs.update({s_name: copy.deepcopy(s) for s_name in names})
            else:
                methods["default"] = method
                method_configs["default"] = s
        return methods, method_configs

    def _get_method(self, name: str) -> dict:
        """Get the method by name

        Parameters
        -------
        name: str
            The hint name

        Returns
        -------
        method: callable
            The method.
        """

        return self._methods.get(name) or self._methods["default"]

    def _get_method_config(self, name: str) -> dict:
        """Get the config for method by name

        Parameters
        -------
        name: str
            The hint name

        Returns
        -------
        method_config: dict
            The method config.
        """

        return self._method_configs.get(name) or self._method_configs["default"]

    @property
    def graph(self):
        return self._graphs[self._graph_id]

    @classmethod
    def tool_type(cls):
        return MSCToolType.BASE

    @classmethod
    def tool_style(cls):
        return "base"


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

    if tool_type == MSCToolType.PRUNE:
        return MSCKey.PRUNER
    if tool_type == MSCToolType.QUANTIZE:
        return MSCKey.QUANTIZER
    if tool_type == MSCToolType.DISTILL:
        return MSCKey.DISTILLER
    if tool_type == MSCToolType.DEBUG:
        return MSCKey.DEBUGGER
    raise TypeError("Unexpected tool type " + str(tool_type))


def add_tool(tool: MSCTool, tool_type: str, tag: str = "main"):
    """Add tool by type and tag

    Parameters
    -------
    tool: MSCTool
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


def create_tool(framework: str, tool_type: str, config: dict, tag: str = "main") -> MSCTool:
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
    tool_cls = msc_utils.get_registered_tool_cls(tool_type, tool_style)
    assert tool_cls, "Can not find tool class for {}:{}".format(tool_type, tool_style)
    impl_style = config.pop("impl_style") if "impl_style" in config else "default"
    impl_cls = msc_utils.get_registered_tool_impl(framework, tool_type, impl_style)
    assert impl_cls, "Can not find implement class for {}:{} @ {}".format(
        tool_type, impl_style, framework
    )
    return add_tool(tool_cls(impl_cls(), **config), tool_type, tag)


def get_tool(tool_type: str, tag: str = "main") -> MSCTool:
    """Get tool by type and tag

    Parameters
    -------
    tool_type: str
        The type of the tool prune| quantize| distill...
    tag: str
        The tag of the tool.

    Returns
    -------
    tool: MSCTool
        The saved tool.
    """

    tool_key = _get_tool_key(tool_type)
    tools = MSCMap.get(tool_key, {})
    return tools.get(tag)


def get_tools(tag: str = "main") -> Iterable[MSCTool]:
    """Get all saved tools by tag

    Parameters
    -------
    tag: str
        The tag of the tool.

    Returns
    -------
    tools: iterable<MSCTool>
        The saved tools.
    """

    for t_type in MSCToolType.all_types():
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
