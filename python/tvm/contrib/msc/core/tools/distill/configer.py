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
"""tvm.contrib.msc.core.tools.distill.configer"""

from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core import utils as msc_utils


class DistillConfiger(object):
    """Configer for distill"""

    def config(self, raw_config: dict = None) -> dict:
        """Get the config

        Parameters
        ----------
        raw_config: dict
            The raw config.

        Returns
        -------
        config: dict
            The update config.
        """

        config = self.update(raw_config) if raw_config else self.get_default()
        return config

    def get_default(self) -> dict:
        """Get the default config"""

        raise NotImplementedError("get_default is not implemented in DistillConfiger")

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


class DefaultDistillConfiger(DistillConfiger):
    """Default configer for distill"""

    def get_default(self) -> dict:
        """Get the default config"""

        return {
            "plan_file": "msc_distiller.json",
            "strategys": [
                {
                    "method": "loss_lp_norm",
                    "op_types": ["loss"],
                },
            ],
        }

    @classmethod
    def tool_type(cls):
        return ToolType.DISTILLER

    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_configer(DefaultDistillConfiger)
