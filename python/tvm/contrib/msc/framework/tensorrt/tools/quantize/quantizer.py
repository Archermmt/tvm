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
"""tvm.contrib.msc.framework.tensorrt.tools.quantize.quantizer"""

import os
import struct
from typing import List, Dict, Tuple

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.tools.tool import ToolType, Strategy
from tvm.contrib.msc.core.tools.quantize import BaseQuantizer
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TensorRTQuantizerFactory(object):
    """Quantizer factory for tensorrt"""

    def create(self, base_cls: BaseQuantizer):
        class Quantizer(base_cls):
            """Adaptive quantizer for tensorrt"""

            def reset(
                self,
                graphs: List[MSCGraph],
                weights: List[Dict[str, tvm.nd.array]],
                cache_dir: msc_utils.MSCDirectory = None,
            ) -> Tuple[List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
                """Reset the tool with graphs and weights

                Parameters
                ----------
                graphs: list<MSCgraph>
                    The msc graphs.
                weights: list<dic<str, tvm.nd.array>>
                    The weights
                cache_dir: MSCDirectory
                    cache path for save/load info

                Returns
                -------
                graphs: list<MSCgraph>
                    The msc graphs.
                weights: list<dic<str, tvm.nd.array>>
                    The weights
                """

                if self._plan:
                    self._use_range = all(
                        info.get("use_range", False) for info in self._plan.values()
                    )
                else:
                    self._use_range = False
                return super().reset(graphs, weights, cache_dir)

            def _execute_before_build(self, codegen_context: dict) -> dict:
                """Execute before model build

                Parameters
                ----------
                codegen_context: dict
                    The context.

                Returns
                ----------
                codegen_context: dict
                    The processed context.
                """

                config_folder = msc_utils.get_config_dir()
                self._range_files = {
                    g.name: config_folder.relpath(g.name + ".range") for g in self._graphs
                }
                self._calibrate_savers = {}
                if self._calibrated:
                    if self._use_range:
                        for graph in self._graphs:
                            if not os.path.isfile(self._range_files[graph.name]):
                                self._plan_to_range(graph, self._range_files[graph.name])
                    else:
                        self._quantized_tensors = set()
                else:
                    calibrate_folder = msc_utils.get_dataset_dir().create_dir("Calibrate")
                    for graph in self._graphs:
                        saver_options = {"input_names": [i.name for i in graph.get_inputs()]}
                        self._calibrate_savers[graph.name] = msc_utils.IODataSaver(
                            calibrate_folder.relpath(graph.name), saver_options
                        )
                super()._execute_before_forward(codegen_context)
                return codegen_context

            def _execute_before_forward(self, step_context: dict) -> dict:
                """Execute before model forward

                Parameters
                ----------
                step_context: dict
                    The context.

                Returns
                ----------
                step_context: dict
                    The processed context.
                """

                if not self._calibrated:
                    saver = self._calibrate_savers[self.graph.name]
                    saver.save_batch(
                        {name: data.asnumpy() for name, data in step_context["datas"].items()}
                    )
                return super()._execute_before_forward(step_context)

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

                if self._calibrated and not self._use_range and name not in self._quantized_tensors:
                    self._quantized_tensors.add(name)
                    plan = self._plan[self._to_tensor_id(name, consumer)]
                    nbits = plan.get("nbits", 8)
                    precision = "DataType::k"
                    if nbits == -1:
                        precision += "FLOAT"
                    if nbits == 8:
                        precision += "INT8"
                    elif nbits == 16:
                        precision += "HALF"
                    else:
                        raise TypeError("nbits {} is not supported".format(nbits))
                    tensor_ctx["processed"].extend(
                        [
                            "{}->setPrecision({})".format(tensor_ctx["producer"], precision),
                            "{0}->setDynamicRange(-{1},{1})".format(
                                tensor_ctx["tensor"], plan["scale"]
                            ),
                        ]
                    )
                return tensor_ctx

            def calibrate(self) -> dict:
                """Calibrate the datas

                Returns
                -------
                plan: dict
                    The calibrated plan.
                """

                for graph in self._graphs:
                    self._range_to_plan(graph, self._range_files[graph.name])
                self._calibrated = True
                return self._plan

            def update_codegen(self, codegen_configs: List[Dict[str, str]]) -> List[Dict[str, str]]:
                """Update the codegen configs
                Parameters
                ----------
                codegen_configs: list<dict<str, str>>
                    The codegen_configs.

                Returns
                -------
                codegen_configs: list<dict<str, str>>
                    The updated codegen_configs.
                """

                if self._calibrated:
                    if self._use_range:
                        for idx, graph in enumerate(self._graphs):
                            if os.path.isfile(self._range_files[graph.name]):
                                codegen_configs[idx].update(
                                    {
                                        "range_file": self._range_files[graph.name],
                                    }
                                )
                else:
                    for idx, graph in enumerate(self._graphs):
                        codegen_configs[idx].update(
                            {
                                "dataset": self._calibrate_savers[graph.name].folder,
                                "range_file": self._range_files[graph.name],
                            }
                        )
                return codegen_configs

            def _plan_to_range(self, graph: MSCGraph, range_file: str, title="MSCCalibrate"):
                """Extract plan config to range_file

                Parameters
                ----------
                plan: dict
                    The plan.
                graph: MSCGraph
                    The graph.
                range_file: str
                    The output range_file path.
                title: str
                    The title of the range file.
                """

                def _scale_to_hex(scale):
                    return hex(struct.unpack("<I", struct.pack("<f", scale / 127))[0])[2:]

                recorded = set()
                with open(range_file, "w") as f:
                    f.write(title + "\n")
                    for name, info in self._plan.items():
                        t_name, _ = self.from_tensor_id(name)
                        if not graph.find_tensor(t_name):
                            continue
                        if t_name not in recorded:
                            f.write("{}: {}\n").format(t_name, _scale_to_hex(info["scale"]))
                            recorded.add(t_name)

            def _range_to_plan(self, graph: MSCGraph, range_file: str):
                """Extract scale in range_file to plan

                Parameters
                ----------
                graph: MSCGraph
                    The graph.
                range_file: str
                    The input range_file path.
                """

                with open(range_file, "r") as f:
                    f.readline()
                    line = f.readline()
                    while line:
                        name, scale = line.split(": ")
                        scale = scale.strip()
                        if scale == "0":
                            value = 0.0
                        else:
                            value = struct.unpack("!f", bytes.fromhex(scale))[0] * 127
                        consumers = graph.find_consumers()
                        if consumers:
                            for c in consumers:
                                self._plan[self.to_tensor_id(name, c.name)] = {
                                    "scale": value,
                                    "use_range": True,
                                }
                        else:
                            self._plan[self.to_tensor_id(name, "exit")] = {
                                "scale": value,
                                "use_range": True,
                            }
                        line = f.readline()

            @classmethod
            def framework(cls):
                return MSCFramework.TENSORRT

        return Quantizer


factory = TensorRTQuantizerFactory()
tools = msc_utils.get_registered_tool_cls(MSCFramework.MSC, ToolType.QUANTIZER, tool_style="all")
for tool in tools.values():
    msc_utils.register_tool_cls(factory.create(tool))
