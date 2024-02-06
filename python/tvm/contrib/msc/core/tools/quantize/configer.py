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
"""tvm.contrib.msc.core.tools.quantize.configer"""

from typing import List, Union

from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core import utils as msc_utils
from .quantizer import QuantizeStage


class QuantizeConfiger(object):
    """Configer for quantize"""

    def config(self, raw_config: dict = None, gym_configs: List[Union[dict, str]] = None) -> dict:
        """Get the config

        Parameters
        ----------
        raw_config: dict
            The raw config.
        gym_configs: list<dict>
            The gym configs

        Returns
        -------
        config: dict
            The update config.
        """

        config = self.update(raw_config) if raw_config else self.get_default()
        if gym_configs:
            config["gym_configs"] = [self.config_gym(g) for g in gym_configs]
        return config

    def get_default(self) -> dict:
        """Get the default config"""

        raise NotImplementedError("get_default is not implemented in QuantizeConfiger")

    def update(self, raw_config: dict) -> dict:
        """Config the raw config

        Parameters
        ----------
        raw_config: dict
            The raw config.

        Returns
        -------
        config: dict
            The update config.
        """

        return raw_config

    def config_gym(self, raw_config: Union[dict, str]) -> dict:
        """Config the gym config

        Parameters
        ----------
        raw_config: dict
            The raw config.

        Returns
        -------
        config: dict
            The update config.
        """

        if isinstance(raw_config, dict):
            return raw_config
        if raw_config == "default":
            return {
                "env": {
                    "executors": {
                        "action_space": {
                            "method": "action_quantize_scale",
                            "start": 0.8,
                            "end": 1.2,
                            "step": 0.1,
                        }
                    },
                },
                "agent": {"agent_type": "search.grid", "executors": {}},
            }
        else:
            raise TypeError("Unexpected gym config " + str(raw_config))

    @classmethod
    def tool_type(cls):
        return ToolType.QUANTIZER


@msc_utils.register_tool_configer
class DefaultQuantizeConfiger(QuantizeConfiger):
    """Default configer for quantize"""

    def get_default(self) -> dict:
        """Get the default config"""

        return {
            "plan_file": "msc_quantizer.json",
            "strategys": [
                {
                    "method": "gather_maxmin",
                    "op_types": ["nn.conv2d", "msc.linear"],
                    "tensor_types": ["input", "output"],
                    "stages": [QuantizeStage.GATHER],
                },
                {
                    "method": "gather_max_per_channel",
                    "op_types": ["nn.conv2d", "msc.linear"],
                    "tensor_types": ["weight"],
                    "stages": [QuantizeStage.GATHER],
                },
                {
                    "method": "calibrate_maxmin",
                    "op_types": ["nn.conv2d", "msc.linear"],
                    "tensor_types": ["input", "output"],
                    "stages": [QuantizeStage.CALIBRATE],
                },
                {
                    "method": "quantize_normal",
                    "op_types": ["nn.conv2d", "msc.linear"],
                    "tensor_types": ["input", "weight"],
                },
                {
                    "method": "dequantize_normal",
                    "op_types": ["nn.conv2d", "msc.linear"],
                    "tensor_types": ["output"],
                },
            ],
        }

    @classmethod
    def tool_style(cls):
        return "default"
