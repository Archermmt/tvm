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
"""tvm.contrib.msc.core.frontend.trace.tarcer"""

import logging
from functools import wraps

import tvm
from .graph import *
from .utils import set_global_tracer, get_global_tracer


class Tracer(object):
    """Tracer of msc

    Parameters
    ----------
    tag: str
        The tag for trace.
    use_cache: bool
        Whether to use cache for trace.
    inputs: list<tuple>
        The input info.
    outputs: list<str>
        The output names.
    scopes: dict
        The config for each scope.
    debug_level: int
        The debug level.
    logger: logging.Logger
        The logger.
    """

    def __init__(
        self,
        dataset: str = "traced_datas",
        use_cache: bool = False,
        inputs: List[tuple] = None,
        outputs: List[str] = None,
        scopes: Dict[str, dict] = None,
        debug_level: int = 0,
        logger: logging.Logger = None,
    ):
        self._dataset, self._savers = msc_utils.msc_dir(dataset), {}
        self._use_cache = use_cache
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._scopes = scopes or {}
        self._debug_level = debug_level
        self._logger = logger or msc_utils.get_global_logger()
        self._parsers = {}
        self._graph = None
        self._logger.info(msc_utils.msg_block(self.mark("SETUP"), self.setup()))

    def setup(self) -> dict:
        """Setup the tracer"""

        return {
            "dataset": self._dataset,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "scopes": self._scopes,
            "debug_level": self._debug_level,
            "use_cache": self._use_cache,
        }

    def reset(self):
        """Reset the tracer"""

        self._graph = TracedGraph("trace")

    def add_input(self, name: str, data: Any) -> TracedTensor:
        """Add input from data

        Parameters
        ----------
        name: str
            The name of input node.
        data
            The data of input.

        Returns
        -------
        tensor: TracedTensor
            The input tensor.
        """

        if self._inputs:
            in_idx = len(self._graph.inputs)
            assert in_idx < len(self._inputs), "input idx {} out of bound {}".format(
                in_idx, self._inputs
            )
            in_info = self._inputs[in_idx]
            if len(in_info) == 3:
                data = {"alias": in_info[0], "shape": in_info[1], "dtype": in_info[2], "obj": data}
            elif len(in_info) == 2:
                data = {"shape": in_info[0], "dtype": in_info[1], "obj": data}
            elif len(in_info) == 1:
                data = {"shape": in_info[0], "obj": data}
            else:
                raise Exception("Unexpected input info " + str(in_info))
        node = self._graph.add_node("input", [], [data], name=name)
        return node.output_at(0)

    def new_scope(self, name: str = None):
        """Start new scope for the graph

        Parameters
        ----------
        name: str
            The name of the scope.
        """

        self._graph.new_scope(name)

    def finalize(self, outputs: List[str]):
        """Finalize the graph

        Parameters
        ----------
        outputs: list<str>
            The output names of the graph.
        """

        self._graph.finalize(outputs)
        if self._outputs:
            assert len(self._outputs) == len(
                outputs
            ), "outputs {} mismatch with required {}".format(len(outputs), self._outputs)
            for o_name, o_alias in zip(outputs, self._outputs):
                self._graph.find_tensor(o_name).set_alias(o_alias)
        info = self._graph.group_up()
        for group in info["groups"]:
            self._save_datas(group)

    def dump(self) -> List[dict]:
        """Dump traced info

        Returns
        -------
        models: list<dict>
            The model configs for msc.
        """

        print("has graph " + str(self._graph))
        for name, saver in self._savers.items():
            saver.finalize()
            self._logger.info(msc_utils.msg_block(self.mark("Datas({})".format(name)), saver.info))
        info = self._graph.group_up()
        model_configs = {k: v for k, v in info.items() if k in ["inputs", "outputs"]}
        model_configs["groups"] = [
            {
                "name": g["name"],
                "inputs": g["inputs"],
                "outputs": g["outputs"],
                "model": self._dump_group(g),
            }
            for g in info["groups"]
        ]
        return model_configs

    def _save_datas(self, group: dict):
        """Save datas for groups

        Parameters
        ----------
        group: dict
            The group info.
        """

        saver = self._savers.get(group["name"])
        if not saver:
            saver_options = {"input_names": group["inputs"], "output_names": group["outputs"]}
            saver = msc_utils.IODataSaver(self._dataset.relpath(group["name"]), saver_options)
            self._savers[group["name"]] = saver
        in_datas = [self._graph.find_tensor(i).data for i in group["inputs"]]
        out_datas = [self._graph.find_tensor(o).data for o in group["outputs"]]
        saver.save_batch(in_datas, out_datas)

    def _dump_group(self, group: dict) -> dict:
        """Dump model info

        Parameters
        ----------
        group: dict
            The group info.

        Returns
        -------
        model: dict
            The model info.
        """

        s_config = self._scopes.get(group["name"], {})
        inputs, outputs = s_config.get("inputs"), s_config.get("outputs")
        if not inputs:

            def _to_info(t_name):
                tensor = self._graph.find_tensor(t_name)
                return (tensor.name, tensor.shape, tensor.dtype)

            inputs = [_to_info(i) for i in group["inputs"]]
        if not outputs:
            outputs = [self._graph.find_tensor(o).name for o in group["outputs"]]
        model, model_ref = None, s_config.get("model_ref")
        if model_ref is None:
            model = self._parse_model(group)
        elif isinstance(model_ref, int):
            node = self._graph.find_node(group["nodes"][model_ref])
            assert node.optype == "model" and node.meta, "Can not get model from node " + str(node)
            model = node.meta
        else:
            raise Exception("Unexpected model_ref " + str(model_ref))
        return {"inputs": inputs, "outputs": outputs, "model": model}

    def _parse_model(self, group: dict) -> tvm.IRModule:
        """Parse the model from nodes

        Parameters
        ----------
        group: dict
            The group info.

        Returns
        -------
        model: IRModule
            The parsed module.
        """

        return None

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

        return "TRACER {}".format(msg)

    @property
    def graph(self):
        return self._graph


def msc_trace(config: dict = None):
    """Wrapper for tracing

    Parameters
    ----------
    config: dict
        The config for trace.
    """

    config = config or {}

    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            tracer = get_global_tracer() or set_global_tracer(Tracer(**config))
            tracer.reset()

            traced_args, traced_kwargs = [], {}
            for idx, arg in enumerate(args):
                if msc_utils.is_array(arg):
                    traced_args.append(tracer.add_input("input_" + str(idx), arg))
                else:
                    traced_args.append(arg)
            for k, v in kwargs.items():
                if msc_utils.is_array(v):
                    traced_kwargs[k] = tracer.add_input(k, arg)
                else:
                    traced_kwargs[k] = v
            tracer.new_scope()
            out_names, res = [], fn(*traced_args, **traced_kwargs)
            if isinstance(res, TracedTensor):
                out_names, res = [res.name], res.data
            elif isinstance(res, (tuple, list)) and all(isinstance(r, TracedTensor) for r in res):
                out_names, res = [r.name for r in res], [r.data for r in res]
            else:
                raise Exception("Unexpected results " + str(res))
            tracer.finalize(out_names)
            return res

        return inner

    return outer


def dump_traced() -> List[dict]:
    """Dump traced info

    Returns
    -------
    models: list<dict>
        The model configs for msc.
    """

    tracer = get_global_tracer()
    assert tracer, "Missing tracer for dump"
    return tracer.dump()
