from torch import nn

class EvaluationNeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.linear_stack = nn.Sequential(
      nn.Conv2d(1, 16, 3, 1, 1),
      nn.ReLU(),
      nn.Conv2d(16, 8, 3, 1, 1),
      nn.ReLU(),
      nn.Conv2d(8, 4, 3, 1, 1),
      nn.ReLU(),

      nn.AvgPool2d(2, 2),

      nn.Flatten(0),

      nn.Linear(64, 128),
      nn.ReLU(),

      nn.Linear(128, 64),
      nn.ReLU(),

      nn.Linear(64, 32),
      nn.ReLU(),

      nn.Linear(32, 1)
    )

  def forward(self, x):
    logits = self.linear_stack(x)

    return logits