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
"""tvm.contrib.msc.framework.tensorrt.codegen.codegen"""

from typing import Dict, Optional

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.codegen import CodeGen
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tensorrt import _ffi_api


def to_tensorrt(
    graph: MSCGraph,
    weights: Optional[Dict[str, tvm.nd.array]] = None,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    output_folder: msc_utils.MSCDirectory = None,
) -> str:
    """Change MSCGraph to TensorRT engine file.

    Parameters
    ----------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    codegen_config: dict
        The config for codegen.
    print_config: dict
        The config for print.
    build_folder: MSCDirectory
        The folder for saving sources and datas.
    export_folder: MSCDirectory
        The folder for saving outputs.

    Returns
    -------
    engine: str
        The engine file.
    """

    build_folder = build_folder or msc_utils.msc_dir(keep_history=False, cleanup=True)

    def _bind_weights(model: str, folder: msc_utils.MSCDirectory) -> str:
        if weights:
            print("should bind weights " + str(weights))
            raise Exception("stop here!!")
        return model

    codegen = CodeGen(
        graph,
        _ffi_api.GetTensorRTSources,
        codegen_config,
        print_config,
        build_folder.create_dir(graph.name),
        output_folder,
        code_format="cpp",
    )
    return codegen.load([], _bind_weights)
