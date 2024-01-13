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
# pylint: disable=unused-argument
"""tvm.contrib.msc.core.runtime.method"""

from typing import Dict, List, Tuple

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core import utils as msc_utils


def update_weights(
    graphs: List[MSCGraph], weights: Dict[str, tvm.nd.array], weights_path: str
) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
    """Update the weights from weights_path

    Parameters
    -------
    graphs: list<MSCGraph>
        The translated graphs
    weights: dict<str, tvm.nd.array>
        The translated weights.
    weights_path: str
        The weights path.

    Returns
    -------
    graphs: list<MSCGraph>
        The updated graphs
    weights: dict<str, tvm.nd.array>
        The updated weights.
    """

    with open(weights_path, "rb") as f:
        new_weights = tvm.runtime.load_param_dict(f.read())
    weights.update({k: v for k, v in new_weights.items() if k in weights})
    return graphs, weights


msc_utils.register_func("update_weights", update_weights)
