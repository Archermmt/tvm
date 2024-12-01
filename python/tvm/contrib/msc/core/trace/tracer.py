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
from tvm import relax
from tvm.contrib.msc.core.frontend import normalize_inputs
from tvm.contrib.msc.core.utils.register import MSCRegistery
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from .utils import set_global_tracer, get_global_tracer
from .graph import *


class Tracer(object):
    """Tracer of msc

    Parameters
    ----------
    dataset: str
        The dataset folder.
    workspace: str
        The workspace
    use_cache: bool
        Whether to use cache for trace.
    inputs: list<tuple>
        The input info.
    outputs: list<str>
        The output names.
    parsers: list<tuple<str, dict>>
        The parsers config.
    scopes: dict
        The config for each scope.
    verbose: str
        The verbose level.
    logger: logging.Logger
        The logger.
    """

    def __init__(
        self,
        inputs: List[tuple] = None,
        outputs: List[str] = None,
        parsers: Dict[str, dict] = None,
        scopes: Dict[str, dict] = None,
        dataset: str = "trace_datas",
        workspace: str = "trace_workspace",
        use_cache: bool = False,
        verbose: str = "info",
        logger: logging.Logger = None,
    ):
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._parser_configs = parsers or ["basic", "numpy"]
        self._scopes = scopes or {}
        self._dataset = msc_utils.msc_dir(dataset, keep_history=use_cache)
        self._workspace = msc_utils.msc_dir(workspace, keep_history=use_cache)
        self._use_cache = use_cache
        self._verbose = verbose
        self._logger = logger
        if not self._logger:
            self._logger = msc_utils.create_file_logger(
                verbose, self._workspace.relpath("TRACER_LOG")
            )
        self._debug_level = 0
        if self._verbose.startswith("debug:"):
            self._debug_level = int(self._verbose.split(":")[1])
        self._logger.info(msc_utils.msg_block(self.mark("SETUP"), self.setup()))

    def setup(self) -> dict:
        """Setup the tracer"""

        self._graph = None
        self._savers = {}
        self._parsers, self._ops_map = {}, {}
        for p_config in self._parser_configs:
            if isinstance(p_config, (tuple, list)):
                m_name, config = p_config
            else:
                m_name, config = p_config, {}
            parser_cls = msc_utils.get_registered_trace_parser(m_name)
            assert parser_cls, "Can not find parser for " + str(m_name)
            self._parsers[m_name] = parser_cls(
                self, config, debug_level=self._debug_level, logger=self._logger
            )
            self._ops_map.update({n: m_name for n in self._parsers[m_name].convert_map})
        self.enable_trace()
        return {
            "dataset": self._dataset,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "parsers": self._parsers,
            "ops": len(self._ops_map),
            "scopes": self._scopes,
            "verbose": self._verbose,
            "use_cache": self._use_cache,
        }

    def reset(self):
        """Reset the tracer"""

        self._graph = TracedGraph("trace")

    def enable_trace(self):
        """Enable tracing"""

        traced_funcs, tensor_attrs = {}, {}
        for m_name, parser in self._parsers.items():
            parser.enable_trace()
            traced_funcs[m_name] = parser.traced_funcs
            for name, info in parser.tensor_attrs.items():
                tensor_attrs.setdefault(name, []).append(info)
        for name in ["shape", "dtype", "device"]:
            tensor_attrs.setdefault(name, []).append(None)
        TracedTensor.traced_attrs = tensor_attrs
        if self._debug_level > 2:
            self._logger.debug(msc_utils.msg_block(self.mark("Traced funcs"), traced_funcs))
            self._logger.debug(msc_utils.msg_block(self.mark("Tensor attrs"), tensor_attrs))

    def disable_trace(self):
        """Disable tracing"""

        for parser in self._parsers.values():
            parser.disable_trace()
        TracedTensor.traced_attrs = {}

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
            if isinstance(in_info, str):
                data = {"alias": in_info, "obj": data}
            elif isinstance(in_info, dict):
                data = msc_utils.update_dict(in_info, {"obj": data})
            elif isinstance(in_info, (tuple, list)):
                if len(in_info) == 3:
                    data = {
                        "alias": in_info[0],
                        "shape": in_info[1],
                        "dtype": in_info[2],
                        "obj": data,
                    }
                elif len(in_info) == 2:
                    data = {"shape": in_info[0], "dtype": in_info[1], "obj": data}
                elif len(in_info) == 1:
                    data = {"shape": in_info[0], "obj": data}
                else:
                    raise Exception("Unexpected input info " + str(in_info))
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
        dumped: dict
            The traced info.
        """

        self.disable_trace()
        print("graph " + str(self._graph))

        datas_info = {}
        for name, saver in self._savers.items():
            saver.finalize()
            datas_info[name] = saver.info
        info = self._graph.group_up()
        g_info = msc_utils.copy_dict(info)

        def _node_des(n_name):
            node = self._graph.find_node(n_name)
            inputs = node.get_inputs()
            des = "{}({})".format(n_name, node.optype)
            if inputs:
                des = ";".join([i.name for i in inputs]) + " -> " + des
            return des

        for g in g_info["groups"]:
            g["nodes"] = [_node_des(n) for n in g["nodes"]]
        self._logger.info(msc_utils.msg_block(self.mark("Datas"), datas_info))
        self._logger.info(msc_utils.msg_block(self.mark("Groups"), g_info))
        dumped = {k: v for k, v in info.items() if k in ["inputs", "outputs"]}
        dumped.update(
            {"dataset": self._dataset, "groups": [self._dump_group(g) for g in info["groups"]]}
        )
        if not self._use_cache:
            self._workspace.destory()
        return dumped

    def destory(self):
        """Clean up the tracer"""

        self._dataset.destory()
        self._workspace.destory()

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
        """Dump group info

        Parameters
        ----------
        group: dict
            The group info.

        Returns
        -------
        group: dict
            The dumped group info.
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
            model, model_type = self._parse_model(group, inputs), MSCFramework.TVM
        elif isinstance(model_ref, int):
            node = self._graph.find_node(group["nodes"][model_ref])
            assert node.optype == "model" and node.meta, "Can not get model from node " + str(node)
            model, model_type = node.meta, node.infer_framework()
        else:
            raise Exception("Unexpected model_ref " + str(model_ref))
        saver = self._savers.get(group["name"])
        assert saver, "Can not find saver for {}, please trace before dump".format(group["name"])
        return {
            "name": group["name"],
            "inputs": inputs,
            "outputs": outputs,
            "model_type": model_type,
            "dataset": saver.folder,
            "model": model,
        }

    def _parse_model(self, group: dict, inputs: List[tuple]) -> tvm.IRModule:
        """Parse the model from nodes

        Parameters
        ----------
        group: dict
            The group info.
        inputs: list
            The inputs info.

        Returns
        -------
        model: IRModule
            The parsed module.
        """

        s_config = self._scopes.get(group["name"], {})
        self.env = {}
        m_inputs, inputs = [], normalize_inputs(inputs)
        for idx, i_info in enumerate(inputs):
            if len(i_info) == 3:
                i_info = {"shape": i_info[1], "dtype": i_info[2]}
            elif len(i_info) == 2:
                i_info = {"shape": i_info[0], "dtype": i_info[1]}
            elif len(i_info) == 1:
                i_info = {"shape": i_info[0], "dtype": "float32"}
            else:
                raise Exception("Unexpected i_info " + str(i_info))
            var = relax.Var(
                "inp_" + str(idx), relax.TensorStructInfo(i_info["shape"], i_info["dtype"])
            )
            m_inputs.append(var)
            self.env[group["inputs"][idx]] = var

        self.block_builder = relax.BlockBuilder()
        with self.block_builder.function(name="main", params=m_inputs, attrs=s_config.get("attrs")):
            for n_name in group["nodes"]:
                node = self._graph.find_node(n_name)
                assert node.optype in self._ops_map, "op {} is not convertable".format(node.optype)
                result = self._parsers[self._ops_map[node.optype]].convert(node)
                outputs = node.get_outputs()
                if len(outputs) == 1 and not node.get_attr("multi_outputs", False):
                    self.env[outputs[0].name] = result
                else:
                    for idx, out in enumerate(outputs):
                        self.env[out.name] = result[idx]
            m_outputs = []
            for o in group["outputs"]:
                assert o in self.env, "Missing output {} in env".format(o)
                m_outputs.append(self.env[o])
            if len(m_outputs) == 1:
                self.block_builder.emit_func_output(m_outputs[0])
            else:
                self.block_builder.emit_func_output(relax.Tuple(m_outputs))
        return self.block_builder.get()

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
                    traced_kwargs[k] = tracer.add_input(k, v)
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
