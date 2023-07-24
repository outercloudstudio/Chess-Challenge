from torch import nn

class EvaluationNeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.linear_stack = nn.Sequential(
      nn.Conv2d(1, 4, 3, 1, 1),
      nn.Tanh(),
      nn.Conv2d(4, 1, 3, 1, 1),
      nn.Tanh(),

      nn.AvgPool2d(2, 2),

      nn.Flatten(1),

      nn.Linear(16, 32),
      nn.Tanh(),

      nn.Linear(32, 16),
      nn.Tanh(),

      nn.Linear(16, 8),
      nn.Tanh(),

      nn.Linear(8, 1)
    )

  def forward(self, x):
    logits = self.linear_stack(x)

    return logits