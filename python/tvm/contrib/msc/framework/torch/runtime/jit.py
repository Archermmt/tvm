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
# pylint: disable=unused-import
"""tvm.contrib.msc.framework.torch.runtime.jit_model"""

from typing import Any, List, Tuple
from functools import partial

import torch
from torch import fx
from torch import _dynamo as dynamo

from tvm.contrib.msc.core.runtime import BaseJIT
from tvm.contrib.msc.core.utils.namespace import MSCFramework


class TorchJIT(BaseJIT):
    """JIT of Torch"""

    def setup(self) -> dict:
        """Setup the jit

        Returns
        -------
        info: dict
            The setup info.
        """

        self._training = self._model.training
        return super().setup()

    def __call__(self, inputs: Any):
        """Call the jit model

        Parameters
        ----------
        inputs:
            The inputs of model.
        """

        if isinstance(inputs, (list, tuple)):
            return self._jit_model(*inputs)
        return self._jit_model(inputs)

    def _build(self, model: Any) -> Any:
        """Build the jit model

        Parameters
        ----------
        model:
            The model.

        Returns
        -------
        jit_model:
            The jit model.
        """

        def _compile(graph_module: fx.GraphModule, example_inputs):
            graph_module = graph_module.train() if self._training else graph_module.eval()
            name = "jit_" + str(len(self._runner_ctxs))
            self._runner_ctxs[name] = {"model": graph_module}
            return partial(self._redirect_forward, runner_name=name)

        dynamo.reset()
        return torch.compile(self._model, backend=_compile)

    def _to_msc_inputs(self, *args, **kwargs) -> List[Tuple[str, Any]]:
        """Change inputs to msc format

        Parameters
        ----------
        args:
            The arguments.
        kwargs:
            The kwargs.

        Returns
        -------
        inputs:
            The msc format inputs.
        """

        assert not kwargs, "TorchJIT do not support kwargs"
        return [("input_" + str(i), d) for i, d in enumerate(args)]

    def _from_msc_outputs(self, outputs: List[Tuple[str, Any]]) -> Any:
        """Change inputs from msc format

        Parameters
        ----------
        outputs: list<(str, tensor)>
            The msc format outputs.

        Returns
        -------
        outputs:
            The framework outputs.
        """

        torch_outputs = [o[1] for o in outputs]
        return torch_outputs[0] if len(torch_outputs) == 1 else torch_outputs

    def _run_ctx(self, runner_ctx: dict, inputs: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """Forward by runner context

        Parameters
        ----------
        runner_ctx: dict
            The runner context
        inputs: list<(str, tensor)>
            The inputs.

        Returns
        -------
        outputs: list<(str, tensor)>
            The outputs.
        """

        if "runner" in runner_ctx:
            outputs = runner_ctx["runner"].run({i[0]: i[1] for i in inputs}, ret_type="native")
        else:
            torch_inputs = [i[1] for i in inputs]
            outputs = runner_ctx["model"](*torch_inputs)
        if isinstance(outputs, (list, tuple)):
            return [("output_" + str(i), o) for i, o in enumerate(outputs)]
        return [("output", outputs)]

    @property
    def framework(self):
        return MSCFramework.TORCH
