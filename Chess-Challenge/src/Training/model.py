from torch import nn

class EvaluationNeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.linear_stack = nn.Sequential(
      nn.Linear(3, 8),
      nn.Tanh(),

      nn.Linear(8, 8),
      nn.Tanh(),

      nn.Linear(8, 8),
      nn.Tanh(),

      nn.Linear(8, 1),
    )

  def forward(self, x):
    logits = self.linear_stack(x)

    return logits