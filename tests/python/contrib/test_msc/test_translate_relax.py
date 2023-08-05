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
import torch
from torch import fx
from torch.nn import Module

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.core.ir import translate
from tvm.contrib.msc.framework.tvm import codegen as tvm_codegen


def verify_model(torch_model, input_info):
    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        expected = from_fx(graph_model, input_info)
    graph, weights = translate.from_relax(expected)
    mod = tvm_codegen.to_relax(graph, weights, codegen_config={"explicit_name": False})
    tvm.ir.assert_structural_equal(mod, expected)


def test_conv1d():
    class Conv1D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv1D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    input_info = [([1, 3, 10], "float32")]
    verify_model(Conv1D1(), input_info)
    verify_model(Conv1D2(), input_info)


def test_conv2d():
    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Conv2D1(), input_info)
    verify_model(Conv2D2(), input_info)


def test_linear():
    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, input):
            return self.linear(input)

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, input):
            return self.linear(input)

    class MatMul1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Dense1(), input_info)
    verify_model(Dense2(), input_info)
    verify_model(MatMul1(), [([10, 10], "float32"), ([10, 10], "float32")])


def test_bmm():
    class BMM(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.bmm(x, y)

    input_info = [((4, 128, 256), "float32"), ((4, 256, 512), "float32")]
    verify_model(BMM(), input_info)


def test_baddbmm():
    class BAddBMM1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    class BAddBMM2(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    input_info = [
        ((4, 128, 512), "float32"),
        ((4, 128, 256), "float32"),
        ((4, 256, 512), "float32"),
    ]
    verify_model(BAddBMM1(), input_info)
    verify_model(BAddBMM2(), input_info)


def test_relu():
    class ReLU(Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, input):
            return self.relu(input)

    class ReLU1(Module):
        def forward(self, input):
            return torch.nn.functional.relu(input)

    input_info = [([10, 10], "float32")]
    verify_model(ReLU(), input_info)
    verify_model(ReLU1(), input_info)


def test_relu6():
    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, input):
            return self.relu6(input)

    input_info = [([10, 10], "float32")]
    verify_model(ReLU6(), input_info)


def test_maxpool2d():
    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3])

        def forward(self, input):
            return self.pool(input)

    class MaxPool2d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, input):
            return self.pool(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(MaxPool2d(), input_info)
    verify_model(MaxPool2d2(), input_info)
    verify_model(MaxPool2d3(), input_info)


def test_avgpool2d():
    class AvgPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    class AvgPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True)

        def forward(self, input):
            return self.pool(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(AvgPool2d(), input_info)
    verify_model(AvgPool2d2(), input_info)


def test_adaptive_avgpool2d():
    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, input):
            return self.pool(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(AdaptiveAvgPool2d0(), input_info)


def test_flatten():
    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, input):
            return self.f(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Flatten(), input_info)
    verify_model(torch.nn.Flatten(2, -1), input_info)


def test_batchnorm2d():
    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, input):
            return self.bn(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(BatchNorm2d(), input_info)


def test_embedding():
    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, input):
            return self.embedding(input)

    verify_model(Embedding(), [([4], "int64")])
    verify_model(Embedding(), [([4, 5], "int64")])


def test_dropout():
    class Dropout1(Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, input):
            return self.dropout(input)

    class Dropout2(Module):
        def forward(self, input):
            return torch.dropout(input, 0.5, train=True)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Dropout1(), input_info)
    verify_model(Dropout2(), input_info)


def test_layernorm():
    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm((10, 10))

        def forward(self, input):
            return self.ln(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(LayerNorm(), input_info)


def test_functional_layernorm():
    class LayerNorm(Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(shape))
            self.bias = torch.nn.Parameter(torch.zeros(shape))

        def forward(self, input):
            return torch.nn.functional.layer_norm(
                input, self.weight.shape, self.weight, self.bias, 1e-5
            )

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(LayerNorm((10, 10)), input_info)


def test_cross_entropy():
    class CrossEntropy1(Module):
        def __init__(self):
            super().__init__()
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    class CrossEntropy2(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2,)))
            self.loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    class CrossEntropy3(Module):
        def __init__(self):
            super().__init__()
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=1, reduction="sum")

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    input_info = [([3, 2], "float32"), ([3], "int32")]
    verify_model(CrossEntropy1(), input_info)
    verify_model(CrossEntropy2(), input_info)
    verify_model(CrossEntropy3(), input_info)


def test_functional_cross_entropy():
    class CrossEntropy(Module):
        def forward(self, logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets)

    input_info = [([3, 10], "float32"), ([3], "int32")]
    verify_model(CrossEntropy(), input_info)


def test_silu():
    class SiLU(Module):
        def __init__(self):
            super().__init__()
            self.silu = torch.nn.SiLU()

        def forward(self, input):
            return self.silu(input)

    class SiLU2(Module):
        def forward(self, input):
            return torch.nn.functional.silu(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(SiLU(), input_info)
    verify_model(SiLU2(), input_info)


def test_groupnorm():
    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.GroupNorm(3, 3)

        def forward(self, input):
            return self.gn(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(GroupNorm(), input_info)


def test_softmax():
    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.sm = torch.nn.Softmax(dim=1)

        def forward(self, input):
            return self.sm(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Softmax(), input_info)


def test_binary():
    input_info1 = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]
    input_info2 = [([1, 3, 10, 10], "float32")]

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    verify_model(Add1(), input_info1)
    verify_model(Add2(), input_info2)

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    verify_model(Sub1(), input_info1)
    verify_model(Sub2(), input_info2)

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    verify_model(Mul1(), input_info1)
    verify_model(Mul2(), input_info2)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    verify_model(TrueDiv1(), input_info1)
    verify_model(TrueDiv2(), input_info2)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    verify_model(FloorDiv1(), input_info1)
    verify_model(FloorDiv2(), input_info2)

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    verify_model(Power1(), input_info1)
    verify_model(Power2(), input_info2)

    # LT
    class LT1(Module):
        def forward(self, lhs, rhs):
            return lhs < rhs

    class LT2(Module):
        def forward(self, lhs):
            return lhs < 1.0

    verify_model(LT1(), input_info1)
    verify_model(LT2(), input_info2)


def test_size():
    class Size(Module):
        def forward(self, input):
            return input.size()

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Size(), input_info)


def test_squeeze():
    class Squeeze1(Module):
        def forward(self, input):
            return input.squeeze(1)

    class Squeeze2(Module):
        def forward(self, input):
            return input.squeeze()

    input_info = [([3, 1, 4, 1], "float32")]
    verify_model(Squeeze1(), input_info)
    verify_model(Squeeze2(), input_info)


def test_unsqueeze():
    class Unsqueeze1(Module):
        def forward(self, input):
            return input.unsqueeze(1)

    class Unsqueeze2(Module):
        def forward(self, input):
            return input.unsqueeze(-1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Unsqueeze1(), input_info)
    verify_model(Unsqueeze2(), input_info)


def test_getattr():
    class GetAttr1(Module):
        def forward(self, input):
            return input.shape

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(GetAttr1(), input_info)


def test_getitem():
    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    verify_model(Slice1(), [([1, 3, 10, 10], "float32")])
    verify_model(Slice2(), [([8, 16], "float32")])


def test_unary():
    input_info = [([1, 3, 10, 10], "float32")]

    # sin
    class Sin(Module):
        def forward(self, input):
            return torch.sin(input)

    verify_model(Sin(), input_info)

    # cos
    class Cos(Module):
        def forward(self, input):
            return torch.cos(input)

    verify_model(Cos(), input_info)

    # exp
    class Exp(Module):
        def forward(self, input):
            return torch.exp(input)

    verify_model(Exp(), input_info)

    # sqrt
    class Sqrt(Module):
        def forward(self, input):
            return torch.sqrt(input)

    verify_model(Sqrt(), input_info)

    # sigmoid
    class Sigmoid(Module):
        def forward(self, input):
            return torch.sigmoid(input)

    verify_model(Sigmoid(), input_info)

    # round
    class Round(Module):
        def forward(self, input):
            return torch.round(input)

    verify_model(Round(), input_info)


def test_gelu():
    class Gelu(Module):
        def forward(self, input):
            return torch.nn.functional.gelu(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Gelu(), input_info)


def test_tanh():
    class Tanh(Module):
        def forward(self, input):
            return torch.tanh(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Tanh(), input_info)


def test_clamp():
    class Clamp(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.1, max=0.5)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Clamp(), input_info)


def test_interpolate():
    class Interpolate(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (5, 5))

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Interpolate(), input_info)


def test_addmm():
    class Addmm(Module):
        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3)

    input_info = [
        ([10, 10], "float32"),
        ([10, 10], "float32"),
        ([10, 10], "float32"),
    ]
    verify_model(Addmm(), input_info)


def test_split():
    class Split(Module):
        def forward(self, input):
            return torch.split(input, 1, dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Split(), input_info)


def test_cumsum():
    class Cumsum(Module):
        def forward(self, input):
            return torch.cumsum(input, dim=1, dtype=torch.int32)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Cumsum(), input_info)


def test_chunk():
    class Chunk(Module):
        def forward(self, input):
            return torch.chunk(input, 3, dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Chunk(), input_info)


def test_inplace_fill():
    class InplaceFill(Module):
        def forward(self, input):
            input.fill_(1.5)
            return input

    verify_model(InplaceFill(), [([10, 10], "float32")])


def test_arange():
    class Arange(Module):
        def forward(self, input):
            return torch.arange(0, 20, dtype=torch.int32)

    verify_model(Arange(), [([10, 10], "float32")])


def test_empty():
    class Empty(Module):
        def forward(self, input):
            return torch.empty((10, 10), dtype=torch.float32)

    verify_model(Empty(), [([10, 10], "float32")])


def test_tensor():
    class Empty1(Module):
        def forward(self, input):
            return torch.tensor(3, dtype=torch.float32)

    class Empty2(Module):
        def forward(self, input):
            return torch.tensor(3)

    verify_model(Empty1(), [([10, 10], "float32")])
    verify_model(Empty2(), [([10, 10], "float32")])


def test_tril():
    class Tril(Module):
        def forward(self, input):
            return torch.tril(input, 1)

    class InplaceTril(Module):
        def forward(self, input):
            input.tril_(1)
            return input

    input_info = [([10, 10], "float32")]
    verify_model(Tril(), input_info)
    verify_model(InplaceTril(), input_info)


def test_triu():
    class Triu(Module):
        def forward(self, input):
            return torch.triu(input, 1)

    class InplaceTriu(Module):
        def forward(self, input):
            input.triu_(1)
            return input

    input_info = [([10, 10], "float32")]
    verify_model(Triu(), input_info)
    verify_model(InplaceTriu(), input_info)


def test_new_ones():
    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    input_info = [([1, 2, 3], "float32")]
    verify_model(NewOnes(), input_info)


def test_expand():
    class Expand(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Expand(), input_info)


def test_reduce():
    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Sum(), input_info)


def test_datatype():
    input_info = [([1, 2, 3, 4], "float32")]

    # float
    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    verify_model(ToFloat(), input_info)

    # half
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    verify_model(ToHalf(), input_info)

    # type
    class Type(Module):
        def forward(self, x):
            return x.type(torch.float32)

    # type
    class TypeFromAttr(Module):
        def forward(self, x):
            return x.type(x.getattr("dtype"))

    # astype
    class AsType(Module):
        def forward(self, x):
            return x.astype(torch.float32)

    verify_model(Type(), input_info)
    verify_model(TypeFromAttr(), input_info)
    verify_model(AsType(), input_info)


def test_permute():
    class Permute(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Permute(), input_info)


def test_reshape():
    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Reshape(), input_info)


def test_transpose():
    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Transpose(), input_info)


def test_view():
    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(View(), input_info)


def test_keep_params():
    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    verify_model(Conv2D1(), [([1, 3, 10, 10], "float32")])


def test_unwrap_unit_return_tuple():
    class Identity(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return (x,)

    verify_model(Identity(), [([256, 256], "float32")])


def test_no_bind_return_tuple():
    class Identity(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return (x, y)

    input_info = [([256, 256], "float32"), ([256, 256], "float32")]
    verify_model(Identity(), input_info)


def test_argmax():
    class Argmax1(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmax(input, dim=-1)

    class Argmax2(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmax(input, dim=-1, keepdim=True)

    verify_model(Argmax1(), [([256, 256], "float32")])
    verify_model(Argmax2(), [([256, 256], "float32")])


def test_argmin():
    class Argmin1(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmin(input)

    class Argmin2(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmin(input, keepdim=True)

    verify_model(Argmin1(), [([256, 256], "float32")])
    verify_model(Argmin2(), [([256, 256], "float32")])


def test_to():
    class To1(Module):
        def forward(self, input):
            return input.to(torch.float16)

    class To2(Module):
        def forward(self, input):
            return input.to("cpu")

    verify_model(To1(), [([256, 256], "float32")])
    verify_model(To2(), [([256, 256], "float32")])


def test_mean():
    class Mean(Module):
        def forward(self, input):
            return input.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, input):
            return input.mean(-1, keepdim=True)

    verify_model(Mean(), [([256, 256], "float32")])
    verify_model(MeanKeepDim(), [([256, 256], "float32")])


def test_rsqrt():
    class Rsqrt(Module):
        def forward(self, input):
            return torch.rsqrt(input)

    verify_model(Rsqrt(), [([256, 256], "float32")])


def test_neg():
    class Neg(Module):
        def forward(self, input):
            return -input

    verify_model(Neg(), [([256, 256], "float32")])


def test_max():
    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    verify_model(Max(), [([256, 256], "float32"), ([256, 256], "float32")])


def test_attention():
    import torch.nn.functional as F

    class Attention1(Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

    class Attention2(Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    input_info = [
        ([32, 8, 128, 64], "float32"),
        ([32, 8, 128, 64], "float32"),
        ([32, 8, 128, 64], "float32"),
    ]
    verify_model(Attention1(), input_info)
    verify_model(Attention2(), input_info)

    class Attention2(Module):
        def forward(self, q, k, v, mask):
            return F.scaled_dot_product_attention(q, k, v, mask)

    verify_model(
        Attention2(),
        [
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 128], "float32"),
        ],
    )


if __name__ == "__main__":
    tvm.testing.main()