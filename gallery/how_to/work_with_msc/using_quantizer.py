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

parser = argparse.ArgumentParser(description="Qauntizer example")
parser.add_argument("--dataset", type=str, help="The folder saving training and testing datas")
parser.add_argument("--compile_type", type=str, default="tvm", help="The compile type of model")
parser.add_argument("--distill", action="store_true", help="Whether to use distiller for quantize")
parser.add_argument("--gym", action="store_true", help="Whether to use gym for quantize")
parser.add_argument("--test_batch", type=int, default=1, help="The batch size for test")
parser.add_argument("--test_iter", type=int, default=-1, help="The iter for test")
parser.add_argument("--train_batch", type=int, default=32, help="The batch size for train")
parser.add_argument("--train_iter", type=int, default=-1, help="The iter for train")

if __name__ == "__main__":
    print("testing the quantizer")
    model = torch
