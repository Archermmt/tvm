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
"""tvm.contrib.msc.core.gym.quantize_env"""

from typing import List
from tvm.contrib.msc.core.tools import BaseTool, ToolType
from tvm.contrib.msc.core import utils as msc_utils
from .base_env import BaseEnv


class QuantizeEnv(BaseEnv):
    """Environment for quantize"""

    def _get_main_tool(self) -> BaseTool:
        """Get the main tool"""

        return self._runner.get_tool(ToolType.QUANTIZER)

    def _update_runner(self, action: float, task_id: int):
        """Update the runner

        Parameters
        ----------
        action: float
            The current action.
        task_id: int
            The current task id.
        """

        raise NotImplementedError("_update_runner is not implemented in BaseEnv")

    def _summary(self, actions: List[float], rewards: List[dict]) -> dict:
        """Summary the final plan

        Parameters
        ----------
        actions: list<float>
            The final actions.
        rewards: list<dict>
            The final rewards.

        Returns
        -------
        plan: dict
            The final plan.
        """

        raise NotImplementedError("_summary is not implemented in BaseEnv")

    @classmethod
    def env_type(cls):
        return msc_utils.MSCStage.QUANTIZE + ".default"


msc_utils.register_gym_env(QuantizeEnv)
