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
# pylint: disable=invalid-name
"""tvm.contrib.msc.framework.tensorrt.transform.transform"""

import tvm
from tvm.relax.transform import _ffi_api as relax_api
from tvm.contrib.msc.core.utils import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


def TransformTensorRT(config: dict = None) -> tvm.ir.transform.Pass:
    """Transform the Function to fit TensorRT.

    Parameters
    ----------
    config: list<int>
        The tensorrt transform config.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """

    config = config or {}
    config.setdefault("version", msc_utils.get_version(MSCFramework.TENSORRT))
    return relax_api.TransformTensorRT(msc_utils.dump_dict(config))  # type: ignore
