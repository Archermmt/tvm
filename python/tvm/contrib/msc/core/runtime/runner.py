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
"""tvm.contrib.msc.core.runtime.runner"""

from typing import Dict, Optional, Any

import tvm


class BaseRunner(object):
    """Basic runner of MSC

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    tools_config: dict
        The config of MSC Tools.
    translate_config: dict
        The config for translate IRModule to MSCGraph.
    codegen_config: dict
        The config for build MSCGraph to runnable model.
    """

    def __init__(
        self,
        mod: tvm.IRModule,
        params: Optional[Dict[str, tvm.nd.array]] = None,
        tools_config: Optional[Dict[str, Any]] = None,
        translate_config: Optional[Dict[str, str]] = None,
        codegen_config: Optional[Dict[str, str]] = None,
    ):
        self._mod = mod
        self._params = params
        self._tools_config = tools_config
        self._translate_config = translate_config
        self._codegen_config = codegen_config
        self._graphs, self._weights = [], []

    def build(self):
        self._graphs, self._weights = self._build(**self._translate_config)
        if self._tools_config:
            raise NotImplementedError("Build runner with tools is not supported")
