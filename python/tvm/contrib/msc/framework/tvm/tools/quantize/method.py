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
"""tvm.contrib.msc.framework.tvm.tools.quantize.method"""

import tvm
from tvm.relax import op as relax_op
from tvm.contrib.msc.core.tools.quantize import QuantizeMethod, BaseQuantizer
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TVMQuantizeMethod(QuantizeMethod):
    """Default quantize method for tvm"""

    @classmethod
    def amplify_data(
        cls,
        data: tvm.relax.Var,
        scale: float,
        min_val: float,
        max_val: float,
        rounding: str = "round",
    ) -> tvm.relax.Var:
        """Amplify the data

        Parameters
        ----------
        data: tvm.relax.Var
            The source data.
        scale: float
            The scale factor
        min_val: float
            The min.
        max_val: float
            The max.
        rounding: str
            The round method

        Returns
        -------
        data: tvm.relax.Var
            The processed data.
        """

        if rounding == "null":
            return relax_op.clip(data * scale, min_val, max_val)
        if rounding == "floor":
            return relax_op.clip(relax_op.floor(data * scale), min_val, max_val)
        if rounding == "ceil":
            return relax_op.clip(relax_op.ceil(data * scale), min_val, max_val)
        if rounding == "round":
            return relax_op.clip(relax_op.round(data * scale), min_val, max_val)
        if rounding == "trunc":
            return relax_op.clip(relax_op.trunc(data * scale), min_val, max_val)
        if rounding == "logic_round":
            data = relax_op.clip(data * scale, min_val, max_val)
            negative_ceil = relax_op.where(
                relax_op.logical_and(data < 0, (data - relax_op.floor(data)) == 0.5),
                relax_op.ceil(data),
                0,
            )
            data = relax_op.where(
                relax_op.logical_and(data < 0, (data - relax_op.floor(data)) == 0.5), 0, data
            )
            data = relax_op.where((data - relax_op.floor(data)) >= 0.5, relax_op.ceil(data), data)
            data = relax_op.where((data - relax_op.floor(data)) < 0.5, relax_op.floor(data), data)
            return data + negative_ceil
        raise TypeError("Unexpected rounding " + str(rounding))

    @classmethod
    def quantize_normal(
        cls,
        quantizer: BaseQuantizer,
        data: tvm.relax.Var,
        name: str,
        consumer: str,
        scale: float,
        nbits: int = 8,
        axis: int = -1,
        sign: bool = True,
        rounding: str = "round",
    ) -> tvm.relax.Var:
        """Calibrate the data by kl_divergence

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        data: tvm.relax.Var
            The source data.
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        scale: float
            The scale factor
        nbits: int
            The number bits for quantize.
        axis: int
            The axis.
        sign: bool
            Whether to use sign.
        rounding str
            The rounding method.

        Returns
        -------
        data: tvm.relax.Var
            The processed tensor.
        """

        valid_range = 2 ** (nbits - int(sign)) - 1
        min_val = -valid_range if sign else 0
        data = cls.amplify_data(data, scale, min_val, valid_range, rounding)
        return data / scale

    @classmethod
    def framework(cls):
        return MSCFramework.TVM


msc_utils.register_tool_method(TVMQuantizeMethod)
