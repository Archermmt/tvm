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
"""
Wrap pytorch model with quantizer.
This example shows how to run PTQ, QAT, PTQ with distill...
Reference for MSC:
https://discuss.tvm.apache.org/t/rfc-unity-msc-introduction-to-multi-system-compiler/15251/5
"""

import argparse
import torch
import torch.optim as optim

from tvm.contrib.msc.pipeline import TorchWrapper
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core import utils as msc_utils
from _resnet import resnet50
from utils import *

"Use resnet50 from https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master"

parser = argparse.ArgumentParser(description="Qauntizer example")
parser.add_argument(
    "--dataset",
    type=str,
    default="/tmp/msc_dataset",
    help="The folder saving training and testing datas",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/tmp/msc_models",
    help="The folder saving training and testing datas",
)
parser.add_argument("--compile_type", type=str, default="tvm", help="The compile type of model")
parser.add_argument("--distill", action="store_true", help="Whether to use distiller for quantize")
parser.add_argument("--gym", action="store_true", help="Whether to use gym for quantize")
parser.add_argument("--test_batch", type=int, default=1, help="The batch size for test")
parser.add_argument("--test_iter", type=int, default=10, help="The iter for test")
parser.add_argument("--calibrate_iter", type=int, default=10, help="The iter for calibration")
parser.add_argument("--train_batch", type=int, default=32, help="The batch size for train")
parser.add_argument("--train_iter", type=int, default=10, help="The iter for train")
parser.add_argument("--train_epoch", type=int, default=1, help="The epoch for train")
args = parser.parse_args()


if __name__ == "__main__":
    trainloader, testloader = get_dataloaders(args.dataset, args.train_batch, args.test_batch)

    def _get_datas():
        for i, (inputs, _) in enumerate(testloader, 0):
            if i >= args.calibrate_iter > 0:
                break
            yield {"input": inputs.detach().cpu().numpy()}

    model = resnet50(pretrained=args.checkpoint)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))

    acc = run_torch_model(model, testloader, max_iter=args.test_iter)
    print("Baseline acc " + str(acc))

    model = TorchWrapper(
        model,
        inputs=[("input", [1, 3, 32, 32], "float32")],
        outputs=["output"],
        compile_type=args.compile_type,
        dataloader=_get_datas,
        quantize_config="default",
        distill_config="default" if args.distill else None,
        gym_configs={ToolType.QUANTIZER: ["default"]} if args.gym else None,
        verbose="debug:1",
    )

    # optimize the model with quantizer(PTQ)
    model.optimize()
    acc = run_torch_model(model, testloader, max_iter=args.test_iter)
    print("PTQ acc " + str(acc))
    for k, v in model.state_dict().items():
        print("ptq weight {} : {}".format(k, msc_utils.inspect_array(v)))

    # train the model with quantizer(QAT)
    optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.08)
    for ep in range(args.train_epoch):
        run_torch_model(model, trainloader, max_iter=args.train_iter, optimizer=optimizer)
        acc = run_torch_model(model, testloader, max_iter=args.test_iter)
        print("QAT[{}] acc: {}".format(ep, acc))

    for k, v in model.state_dict().items():
        print("qat weight {} : {}".format(k, msc_utils.inspect_array(v)))

    # compile the model
    model.compile()
    acc = run_tvm_model(model, testloader, max_iter=args.test_iter)
    print("Compiled acc " + str(acc))
