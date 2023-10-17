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

""" Test SetExprLayout Pass. """

import numpy as np
import torch
from torch import fx

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.framework.tvm.runtime import RelaxRunner
from tvm.contrib.msc.core import utils as msc_utils


def _get_module_from_torch(name):
    """Get model from torch vision and parse to relax"""
    # pylint: disable=import-outside-toplevel
    try:
        import torchvision

        return getattr(torchvision.models, name)()
    except:  # pylint: disable=bare-except
        print("please install torchvision package")
        return None


def _test_relax_runner(device):
    torch_model = _get_module_from_torch("resnet50")
    if torch_model:
        workspace = msc_utils.set_workspace("msc_test")
        graph_model = fx.symbolic_trace(torch_model)
        input_info = [([1, 3, 224, 224], "float32")]
        datas = [np.random.rand(*i[0]).astype(i[1]) for i in input_info]
        torch_datas = [torch.from_numpy(d) for d in datas]
        with torch.no_grad():
            golden = torch_model(*torch_datas)
            mod = from_fx(graph_model, input_info)
        runner = RelaxRunner(mod, device=device)
        runner.build()
        outputs = runner.run(datas, ret_type="list")
        golden = [msc_utils.cast_array(golden)]
        for gol_r, out_r in zip(golden, outputs):
            tvm.testing.assert_allclose(gol_r, out_r, atol=1e-3, rtol=1e-3)
        workspace.destory()


def test_relax_runner_cpu():
    """Test runner for relax on cpu"""

    _test_relax_runner("cpu")


@tvm.testing.requires_gpu
def test_relax_runner_gpu():
    """Test runner for relax on gpu"""

    _test_relax_runner("gpu")


if __name__ == "__main__":
    tvm.testing.main()
