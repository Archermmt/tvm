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
Compile VLA(Vision-Language-Action) model on given target.
This example use model "openvla/openvla-7b" from huggingface and
test code from https://github.com/openvla/openvla

Please download vla model by
huggingface-cli download --resume-download openvla/openvla-7b --local-dir openvla-7b
"""

import argparse
import requests
from io import BytesIO
from PIL import Image
from functools import partial
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from torch import fx
from torch import _dynamo as dynamo

from tvm.contrib.msc.pipeline import TorchWrapper
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils

parser = argparse.ArgumentParser(description="MSC compile vla example")
parser.add_argument("--model", type=str, default="openvla-7b", help="The model path")
parser.add_argument(
    "--image",
    type=str,
    default="https://d15shllkswkct0.cloudfront.net/wp-content/blogs.dir/1/files/2024/06/openvla-model-arm-carrot.png",
    help="The test image paath or url",
)
parser.add_argument(
    "--v_target", type=str, default="torch", help="The compile type of vision model"
)
parser.add_argument(
    "--l_target", type=str, default="torch", help="The compile type of language model"
)
parser.add_argument(
    "--verbose", type=str, default="info", help="The verbose level, info|debug:1,2,3|critical"
)
parser.add_argument(
    "--device", type=str, default="cpu", help="The device for baseline and compile check cuda|cpu"
)
parser.add_argument(
    "--dtype", type=str, default="float32", help="The datatype for baseline and compile"
)
args = parser.parse_args()


def get_config(example_inputs, compile_type):
    inputs, datas = [], {}
    for i in example_inputs:
        if not isinstance(i, torch.Tensor):
            continue
        i_name = "input_" + str(len(inputs))
        datas[i_name] = msc_utils.cast_array(i)
        inputs.append((i_name, datas[i_name].shape, datas[i_name].dtype.name))

    return TorchWrapper.create_config(
        inputs=inputs,
        outputs=["output"],
        compile_type=compile_type,
        dataset={MSCStage.PREPARE: {"loader": datas}},
        verbose=args.verbose,
        skip_config={
            MSCStage.BASELINE: "check",
            MSCStage.OPTIMIZE: "stage",
            MSCStage.COMPILE: "check",
        },
        run_config={"all": {"device": args.device}},
    )


def wrap_forward(*inputs, model=None):
    t_inputs = [i.to(torch.device(args.device)) for i in inputs if isinstance(i, torch.Tensor)]
    outputs = model(*t_inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    return [o.to(torch.device("cpu")) for o in outputs]


if __name__ == "__main__":
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model, attn_implementation="sdpa", torch_dtype=torch.float32, trust_remote_code=True
    )

    """
    def _capture_vision(graph_module: fx.GraphModule, example_inputs):
        workspace = msc_utils.msc_dir("msc_workspace").create_dir("vision").path
        model = TorchWrapper(
            graph_module.eval(), get_config(example_inputs, args.v_target), workspace=workspace
        )
        model.compile()
        return partial(wrap_forward, model=model)

    def _capture_projector(graph_module: fx.GraphModule, example_inputs):
        workspace = msc_utils.msc_dir("msc_workspace").create_dir("projector").path
        model = TorchWrapper(
            graph_module.eval(), get_config(example_inputs, args.v_target), workspace=workspace
        )
        model.compile()
        return partial(wrap_forward, model=model)

    def _capture_language(graph_module: fx.GraphModule, example_inputs):
        print("[TMINFO] _capture_language {}".format(graph_module))
        workspace = msc_utils.msc_dir("msc_workspace").create_dir("language").path
        model = TorchWrapper(
            graph_module.eval(), get_config(example_inputs, args.l_target), workspace=workspace
        )
        model.compile()
        return partial(wrap_forward, model=model)

    dynamo.reset()
    vla.vision_backbone = torch.compile(vla.vision_backbone, backend=_capture_vision, dynamic=False)
    vla.projector = torch.compile(vla.projector, backend=_capture_projector, dynamic=False)
    # vla.language_model = torch.compile(vla.language_model, backend=_capture_language, dynamic=True)
    """

    # test with image
    response = requests.get(args.image)
    image = Image.open(BytesIO(response.content))

    prompt = "In: What action should the robot take to place the bottle?\nOut:"
    inputs = processor(prompt, image)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    print("action " + str(action))
