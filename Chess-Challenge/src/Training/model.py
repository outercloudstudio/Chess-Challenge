from torch import nn

class EvaluationNeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.flatten = nn.Flatten()
    self.linear_stack = nn.Sequential(
      nn.Linear(64, 16),
      nn.Tanh(),

      nn.Linear(16, 16),
      nn.Tanh(),

      nn.Linear(16, 16),
      nn.Tanh(),

      nn.Linear(16, 1)
    )

  def forward(self, x):
    logits = self.linear_stack(x)

    return logits.squeeze()