import os
import numpy as np
from typing import List, Dict
import tvm
from tvm.contrib.msc.core import utils as msc_utils
import torch
from torch import nn
from torch.nn import functional

# Define the helpers
def process_tensor(tensor: torch.Tensor, name: str, consumer: str) -> torch.Tensor:
  return tensor

def load_data(name: str, shape: List[int], dtype: str) -> np.ndarray:
  path = os.path.join("baseline", name + ".bin")
  if os.path.isfile(path):
    data = np.fromfile(path, dtype=dtype).reshape(shape)
  else:
    data = np.ones((shape)).astype(dtype)
  return data


# Define the graph
class main(torch.nn.Module):
  def __init__(self: torch.nn.Module) -> torch.nn.Module:
    super(main, self).__init__()
    # msc.conv1d_bias(msc.conv1d_bias): <res_0> -> <res_1>
    self.msc_conv1d_bias = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=[7], stride=[1], dilation=[1], groups=1, bias=True)

  def forward(self: torch.nn.Module, res_0: torch.Tensor) -> List[torch.Tensor]:
    # msc.conv1d_bias(msc.conv1d_bias): <res_0> -> <res_1>
    res_1 = self.msc_conv1d_bias(res_0)
    outputs = res_1
    return outputs


# Define the test
if __name__ == "__main__":
  # Prepare test datas
  inputs = {}
  golden = {}
  inputs["input0"] = load_data("input0", [1, 3, 10], "float32")
  golden["msc.conv1d_bias"] = load_data("msc.conv1d_bias", [1, 6, 4], "float32")
  # Build and inference the graph
  # Build Model
  model = main()
  # Load weights
  weights = torch.load("main.pth")
  model.load_state_dict(weights)
  res_0 = torch.from_numpy(inputs["input0"])
  outputs = model(res_0)
  msc_utils.compare_arrays(golden, outputs, verbose="detail")