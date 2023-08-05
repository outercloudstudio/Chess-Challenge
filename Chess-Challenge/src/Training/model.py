from torch import nn

class EvaluationNeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.linear_stack = nn.Sequential(
      nn.Conv2d(6, 32, 3, 1, 1),
      nn.Tanh(),

      nn.Conv2d(32, 8, 3, 1, 1),
      nn.Tanh(),

      nn.Conv2d(8, 1, 3, 1, 1),
      nn.Tanh(),

      nn.Flatten(),

      nn.Linear(64, 128),
      nn.Tanh(),

      nn.Linear(128, 64),
      nn.Tanh(),

      nn.Linear(64, 16),
      nn.Tanh(),

      nn.Linear(16, 1),
    )

  def forward(self, x):
    logits = self.linear_stack(x)

    return logits