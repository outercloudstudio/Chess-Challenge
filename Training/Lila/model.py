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
      nn.Linear(6 * 9, 8),
      nn.Tanh(),

      nn.Linear(8, 8),
      nn.Tanh(),

      nn.Linear(8, 1),
      nn.Tanh(),
    )

    self.evaluationStack = nn.Sequential(
      nn.Linear(37, 16),
      nn.Tanh(),

      nn.Linear(16, 16),
      nn.Tanh(),

      nn.Linear(16, 1),
      nn.Tanh(),
    )

  def forward(self, inp):
    inp = inp.view(64 * 6 + 1)

    vision = torch.zeros(37, dtype=torch.float32).to(device)

    for i in range(36):
      x = i // 6
      y = i % 6

      squareTensors = []

      for offsetX in range(3):
        for offsetY in range(3):
          squareTensors.append(inp[(x + offsetX) * 8 * 6 + (y + offsetY) * 6: (x + offsetX) * 8 * 6 + (y + offsetY) * 6 + 6])

      squareTensor = torch.cat(squareTensors).to(device)

      vision[i] = self.sightStack(squareTensor)

    vision[36] = inp[384]

    return self.evaluationStack(vision)