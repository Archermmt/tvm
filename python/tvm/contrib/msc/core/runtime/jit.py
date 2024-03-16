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
"""tvm.contrib.msc.core.runtime.jit_model"""

import logging
from typing import Any, List, Tuple

from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from .runner import BaseRunner


class BaseJIT(object):
    """Base Just-In-Time compile for msc

    Parameters
    ----------
    model:
        The model to be jit compile.
    hooks: dict
        The hooks for runners.
    logger: logging.Logger
        The logger
    """

    def __init__(
        self,
        model: Any,
        hooks: dict = None,
        logger: logging.Logger = None,
    ):
        self._model = model
        self._jit_model = model
        self._hooks = hooks or {}
        self._runner_ctxs = {}
        self._training, self._trained = False, False
        self._logger = logger or msc_utils.get_global_logger()
        self._logger.info(msc_utils.msg_block(self.jit_mark("SETUP"), self.setup()))

    def setup(self) -> dict:
        """Setup the jit

        Returns
        -------
        info: dict
            The setup info.
        """

        return {"hooks": self._hooks, "training": self._training}

    def __call__(self, inputs: Any):
        """Call the jit model

        Parameters
        ----------
        inputs:
            The inputs of model.
        """

        raise NotImplementedError("__call__ is not implemented in " + str(self.__class__))

    def set_runner(self, runner_name: str, runner: BaseRunner):
        """Set runner in runner ctx

        Parameters
        ----------
        runner_name: str
            The runner name.
        runner: BaseRunner
            The runner.
        """

        self.get_runner_ctx(runner_name)["runner"] = runner

    def build(self):
        """Build the jit model"""

        self._jit_model = self._build(self._model)

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

        raise NotImplementedError("_build is not implemented in " + str(self.__class__))

    def _redirect_forward(self, *args, runner_name: str = "worker", **kwargs) -> Any:
        """Redirect forward of model

        Parameters
        ----------
        args:
            The arguments.
        runner_name: str
            The runner name.
        kwargs:
            The kwargs.

        Returns
        -------
        outputs:
            The outputs.
        """

        assert runner_name in self._runner_ctxs, "Failed to create runner " + runner_name
        inputs = self._to_msc_inputs(*args, **kwargs)
        for hook in self._hooks.get("pre_forward", []):
            hook(runner_name, inputs)
        outputs = self._run_ctx(self.get_runner_ctx(runner_name), inputs)
        for hook in self._hooks.get("post_forward", []):
            outputs = hook(runner_name, outputs)
        return self._from_msc_outputs(outputs)

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

        raise NotImplementedError("_to_msc_inputs is not implemented in " + str(self.__class__))

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

        raise NotImplementedError("_from_msc_outputs is not implemented in " + str(self.__class__))

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

        raise NotImplementedError("_run_ctx is not implemented in " + str(self.__class__))

    def get_runner_ctx(self, runner_name: str) -> dict:
        """Get the runner context

        Parameters
        ----------
        runner_name: str
            The runner name

        Returns
        -------
        runner_cts: dict
            The runner context.
        """

        assert runner_name in self._runner_ctxs, "Can not finc runner_context " + str(runner_name)
        return self._runner_ctxs[runner_name]

    def train(self):
        """Change status to train"""

        if not self._training:
            self._training = True
            for runner_ctx in self._runner_ctxs.values():
                if "runner" in runner_ctx:
                    runner_ctx["runner"].train()

    def eval(self):
        """Change status to eval"""

        if self._training:
            self._training, self._trained = False, True
            for runner_ctx in self._runner_ctxs.values():
                if "runner" in runner_ctx:
                    runner_ctx["runner"].eval()

    def jit_mark(self, msg: str):
        """Mark the message with jit info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "JIT({}) {}".format(self.framework, msg)

    @property
    def trained(self):
        return self._trained

    @property
    def jit_model(self):
        return self._jit_model

    @property
    def framework(self):
        return MSCFramework.MSC
