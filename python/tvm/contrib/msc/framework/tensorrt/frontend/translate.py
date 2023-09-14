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
"""tvm.contrib.msc.framework.torch.frontend.translate"""

from typing import Dict, Optional, Tuple, List

import tvm
from tvm import relax
from tvm.relax.transform import BindParams
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.contrib.msc.core import transform as msc_transform
from tvm.contrib.msc.core.ir import MSCGraph, from_relax_func
from tvm.contrib.msc.framework.tensorrt import transform as trt_transform


def partition_for_tensorrt(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.array]] = None,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
) -> Tuple[tvm.IRModule, List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
    """Partition module to tensorrt sub functions.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    trans_config: dict
        The config for transfrorm IRModule.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    build_config: dict
        The config for build MSCGraph.

    Returns
    -------
    mod: IRModule
        The IRModule of partitioned relax.
    graphs: list<MSCGraph>
        The MSCGraphs.
    weights: list
        The weights for MSCGraph.
    """

    trans_config = trans_config or {}
    if params:
        mod = BindParams("main", params)(mod)

    patterns = get_patterns_with_prefix("msc_tensorrt")
    mod = tvm.transform.Sequential(
        [
            msc_transform.SetExprName(),
            trt_transform.TransformTensorRT(),
            relax.transform.FoldConstant(),
            msc_transform.SetExprLayout(trans_config.get("allow_layout_missing", True)),
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
        ]
    )(mod)

    def _is_tensorrt_func(func):
        if "Codegen" not in func.attrs:
            return False
        return func.attrs["Codegen"] == "msc_tensorrt"

    tensorrt_funcs = [func for _, func in mod.functions.items() if _is_tensorrt_func(func)]
    msc_graphs, msc_weights = [], []
    for idx, func in enumerate(tensorrt_funcs):
        graph, weights = from_relax_func(
            func, "msc_tensorrt_" + str(idx), build_config=build_config
        )
        msc_graphs.append(graph)
        msc_weights.append(weights)
        print("get graph[{}]: {}".format(idx, graph))
        print("weitgh " + str(weights))

    """
    mod = tvm.transform.Sequential(
        [
            relax.transform.RunCodegen(),
        ]
    )(mod)
    print("[TMINFO] compiled mod " + str(mod))
    """
    raise Exception("stop here!!")
    return mod, msc_graphs, msc_weights
