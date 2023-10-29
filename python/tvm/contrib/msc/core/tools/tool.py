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
from typing import List, Iterable, Any, Tuple

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
    graphs: list<MSCGraph>
        The msc graphs
    runtime_config: str
        The runtime config file path.
    strategy: dict
        The strategy of the tool.
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
        graphs: List[MSCGraph],
        runtime_config: str,
        strategy: dict,
        options: dict = None,
        verbose_step: int = 50,
        logger: logging.Logger = None,
    ):
        self._tool_impl = tool_impl
        self._graphs = graphs
        if os.path.isfile(runtime_config):
            self._runtime_config = msc_utils.load_dict(runtime_config)
        else:
            self._runtime_config = {}
        self._strategy = self._parse_stratgey(strategy)
        self._verbose_step = verbose_step
        self._logger = logger or msc_utils.get_global_logger()
        self.setup(options)
        init_title = "{}.Init ({} on {})".format(
            self.tool_type(), self.tool_style(), self._tool_impl.framework()
        )
        init_info = {
            "strategy": self._strategy,
            "runtime_config": self._runtime_config,
            "options": self._options,
            "verbose_step": self._verbose_step,
        }
        self._logger.info(msc_utils.msg_block(init_title, init_info))
        self.reset()

    def setup(self, options: dict):
        """Setup the tool

        Parameters
        ----------
        options: dict
            The options for setup the tool
        """

        self._options = options or {}
        self._enabled = True
        self._is_training = False
        self._graph_id = 0
        self._tool_impl.setup(options)

    def reset(self):
        """Reset the tool"""

        self._forward_cnt = 0

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
            self._graph_id = kwargs.get("graph_id", 0)
            self._logger.debug(
                "<{}> before build graph[{}]({})".format(
                    self.tool_type, self._graph_id, "train" if self._is_training else "eval"
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
                    self.tool_type, self._graph_id, "train" if self._is_training else "eval"
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
            if self.should_log():
                self._logger.debug(
                    "<{}> start graph[{}] forward[{}]".format(
                        self.tool_type, self._graph_id, self._forward_cnt
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
                        self.tool_type, self._graph_id, self._forward_cnt
                    )
                )
            self._forward_cnt += 1
        return output

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

    def _parse_stratgey(self, strategy: dict):
        """Parse the strategy to get valid strategy

        Parameters
        -------
        strategy: dict
            The given strategy

        Returns
        -------
        valid_stratgey: dict
            The parsed strategy.
        """

        valid_stratgey = {}
        assert isinstance(strategy, list) and all(
            isinstance(s, dict) for s in strategy
        ), "Strategy should be given as list of dict"
        for s in strategy:
            if "op_types" in s:
                valid_stratgey.update({op_type: copy.deepcopy(s) for op_type in s.pop("op_types")})
            elif "op_names" in s:
                valid_stratgey.update({op_name: copy.deepcopy(s) for op_name in s.pop("op_names")})
            else:
                valid_stratgey["default"] = s
        return valid_stratgey

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

    if tool_type == "prune":
        return MSCKey.PRUNER
    if tool_type == "quantize":
        return MSCKey.QUANTIZER
    if tool_type == "distill":
        return MSCKey.DISTILLER
    if tool_type == "record":
        return MSCKey.RECORDER
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

    for t_type in [MSCToolType.PRUNE, MSCToolType.QUANTIZE, MSCToolType.DISTILL, MSCToolType.DEBUG]:
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
                    output = tool.execute_after_build(*args, **kwargs)
                elif stage == "forward":
                    output = tool.execute_after_forward(*args, **kwargs)
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
