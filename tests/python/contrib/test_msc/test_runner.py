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


def test_relax_runner():
    """Test runner for relax"""

    torch_model = _get_module_from_torch("resnet50")
    if torch_model:
        msc_utils.set_workspace("msc_test")
        graph_model = fx.symbolic_trace(torch_model)
        input_info = [([1, 3, 224, 224], "float32")]
        with torch.no_grad():
            mod = from_fx(graph_model, input_info)
        runner = RelaxRunner(mod)
        print("runner " + str(runner))


if __name__ == "__main__":
    # tvm.testing.main()
    test_relax_runner()
