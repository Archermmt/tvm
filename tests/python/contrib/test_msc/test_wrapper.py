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

""" Test Wrappers in MSC. """

import pytest
import numpy as np
import torch

import tvm.testing
from tvm.contrib.msc.pipeline import TorchManager
from tvm.contrib.msc.core.utils.namespace import MSCFramework


def _get_torch_model(name, is_training=False):
    """Get model from torch vision"""
    # pylint: disable=import-outside-toplevel
    try:
        import torchvision

        model = getattr(torchvision.models, name)(pretrained=True)
        if is_training:
            model = model.train()
        else:
            model = model.eval()
        return model
    except:  # pylint: disable=bare-except
        print("please install torchvision package")
        return None


@pytest.mark.parametrize("compile_type", [MSCFramework.TORCH, MSCFramework.TVM])
def test_torch_wrapper(compile_type):
    """Test wrapper for torch"""

    torch_model = _get_torch_model("resnet50")
    if torch_model:
        datas = [np.random.rand(1, 3, 224, 224).astype("float32") for _ in range(10)]
        datas = [torch.from_numpy(d) for d in datas]
        if torch.cuda.is_available():
            torch_model = torch_model.to(torch.device("cuda:0"))
            datas = [d.to(torch.device("cuda:0")) for d in datas]

        def load_datas():
            for d in datas:
                yield d

        torch_model = TorchWrapper(
            torch_model,
            compile_type,
            inputs=[["input_0", [1, 3, 224, 224], "float32"]],
            outputs=["output"],
            data_loader=load_datas,
            quantize_strategy="default",
        )
        torch_model.optimize()

        torch_model.compile()


if __name__ == "__main__":
    # tvm.testing.main()
    test_torch_wrapper(MSCFramework.TORCH)
