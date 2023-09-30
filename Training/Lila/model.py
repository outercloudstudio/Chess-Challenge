import torch

from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class LilaModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.sightStack = nn.Sequential(
      nn.Linear(6 * 9, 12),
      CRelu(),

      nn.Linear(12, 12),
      CRelu(),

      nn.Linear(12, 4),
      CRelu(),
    )

    self.evaluationStack = nn.Sequential(
      nn.Linear(36 * 4 + 1, 32),
      CRelu(),

      nn.Linear(32, 32),
      CRelu(),

      nn.Linear(32, 1),
      CRelu(),
    )

  def forward(self, inp):
    inp = inp.view(64 * 6 + 1)

    vision = torch.zeros(36 * 4 + 1, dtype=torch.float32).to(device)

    for i in range(36):
      x = i // 6
      y = i % 6

      squareTensors = []

      for offsetX in range(3):
        for offsetY in range(3):
          squareTensors.append(inp[(x + offsetX) * 8 * 6 + (y + offsetY) * 6: (x + offsetX) * 8 * 6 + (y + offsetY) * 6 + 6])

      squareTensor = torch.cat(squareTensors).to(device)

      vision[i * 4] = self.sightStack(squareTensor)[0]
      vision[i * 4 + 1] = self.sightStack(squareTensor)[1]
      vision[i * 4 + 2] = self.sightStack(squareTensor)[2]
      vision[i * 4 + 3] = self.sightStack(squareTensor)[3]

    vision[36 * 4] = inp[384]

    return self.evaluationStack(vision)
  
class CRelu(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x):
    return torch.clamp(x, min=-1, max=1)