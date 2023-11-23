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
"""tvm.contrib.msc.framework.tensorrt.tools.track.tracker"""

from typing import Dict

from tvm.contrib.msc.core.tools.tool import ToolType, Strategy
from tvm.contrib.msc.core.tools.track import BaseTracker
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TensorRTTrackerFactory(object):
    """Tracker factory for tensorrt"""

    def create(self, base_cls: BaseTracker):
        class tracker(base_cls):
            """Adaptive tracker for tensorrt"""

            def _process_tensor(
                self, tensor_ctx: Dict[str, str], name: str, consumer: str, strategy: Strategy
            ) -> Dict[str, str]:
                """Process tensor

                Parameters
                -------
                tensor_ctx: dict<str, str>
                    Tensor describe items.
                name: str
                    The name of the tensor.
                consumer: str
                    The name of the consumer.
                strategy: Strategy
                    The strategy for the tensor

                Returns
                -------
                tensor_ctx: dict<str, str>
                    Tensor items with processed.
                """

                if self.is_weight(name):
                    return self._track_tensor(self.get_data(name), name, strategy)
                print("has tensor_ctx " + str(tensor_ctx))
                print("name {}, with consumer {}".format(name, consumer))
                raise Exception("stop here!!")
                return tensor_ctx

            @classmethod
            def framework(cls):
                return MSCFramework.TENSORRT

        return tracker


factory = TensorRTTrackerFactory()
tools = msc_utils.get_registered_tool_cls(MSCFramework.MSC, ToolType.TRACK, tool_style="all")
for tool in tools.values():
    msc_utils.register_tool_cls(factory.create(tool))
