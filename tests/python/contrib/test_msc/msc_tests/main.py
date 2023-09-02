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
    # const(constant): <res_1>
    data = torch.Tensor(20)
    self.const = nn.Parameter(data)

  def forward(self: torch.nn.Module, res_0: torch.Tensor) -> List[torch.Tensor]:
    # const(constant): <res_1>
    res_1 = self.const
    outputs = res_1
    return outputs


# Define the test
if __name__ == "__main__":
  # Prepare test datas
  inputs = {}
  golden = {}
  inputs["inp_0"] = load_data("inp_0", [10, 10], "float32")
  golden["const"] = load_data("const", [20], "int32")
  # Build and inference the graph
  # Build Model
  model = main()
  # Load weights
  weights = torch.load("main.pth")
  model.load_state_dict(weights)
  res_0 = torch.from_numpy(inputs["inp_0"])
  outputs = model(res_0)
  msc_utils.compare_arrays(golden, outputs, verbose="detail")