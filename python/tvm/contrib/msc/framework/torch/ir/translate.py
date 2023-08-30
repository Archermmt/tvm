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
"""tvm.contrib.msc.framework.torch.ir.translate"""

import numpy as np
from typing import Dict, Optional, Tuple, List

import torch
import tvm
from tvm.relax.frontend.torch import from_fx
from tvm.relay.frontend import from_pytorch

from tvm.contrib.msc.core.ir.graph import MSCGraph
from tvm.contrib.msc.core.ir.translate import from_relax, from_relay
from tvm.contrib.msc.core.codegen import CodeGen
from tvm.contrib.msc.framework.torch import _ffi_api


def from_torch(
    model: torch.nn.Module,
    input_info: List[Tuple[Tuple[int], str]],
    input_names: List[str] = None,
    via_relax: bool = True,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
    opt_config: Optional[Dict[str, str]] = None,
) -> Tuple[MSCGraph, Dict[str, tvm.nd.array]]:
    """Change torch nn.Module to MSCGraph.

    Parameters
    ----------
    model: torch.nn.Module
        The torch module.
    input_info: list
        The input info in format [(shape, dtype)].
    input_names: list<str>
        The input names.
    via_relax: bool
        Whether translate torch to relax.
    trans_config: dict
        The config for transfrorm IRModule.
    build_config: dict
        The config for build MSCGraph.
    opt_config: dict
        The config for optimize the relay before translate.

    Returns
    -------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The weights from the IRModule.
    """

    trans_config = trans_config or {}
    build_config = build_config or {}
    if via_relax:
        graph_model, params = torch.fx.symbolic_trace(model), None
        with torch.no_grad():
            relax_mod = from_fx(graph_model, input_info)
    else:
        datas = [np.random.rand(*i[0]).astype(i[1]) for i in input_info]
        torch_datas = [torch.from_numpy(i) for i in datas]
        scripted_model = torch.jit.trace(model, tuple(torch_datas)).eval()
        if input_names:
            assert len(input_names) == len(
                input_info
            ), "input_names {} length mismatch with input_info {}".format(input_names, input_info)
            shape_list = [(i_name, i_info) for i_name, i_info in zip(input_names, input_info)]
        else:
            shape_list = [("input" + str(idx), i_info) for idx, i_info in enumerate(input_info)]
        relay_mod, params = from_pytorch(scripted_model, shape_list)
        graph, params = from_relay(
            relay_mod,
            params,
            trans_config=trans_config,
            build_config=build_config,
            opt_config=opt_config,
        )
        source_getter = tvm.get_global_func("msc.framework.tvm.GetRelaxSources")
        codegen = CodeGen(graph, source_getter)
        inputs = [
            tvm.relax.Var(i.alias, tvm.relax.TensorStructInfo(i.get_shape(), i.dtype_name))
            for i in graph.get_inputs()
        ]
        relax_mod = codegen.load(inputs)
    graph, weights = from_relax(
        relax_mod, params, trans_config=trans_config, build_config=build_config
    )
    # set alias for weights
    for node in graph.get_nodes():
        for ref, weight in node.get_weights().items():
            if node.optype == "constant":
                alias = node.name.replace(".", "_")
            else:
                alias = node.name.replace(".", "_") + "." + ref
            weight.set_alias(alias)
    return graph, weights
