from torch import nn

class EvaluationNeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.linear_stack = nn.Sequential(
      nn.Linear(17, 32),
      nn.Sigmoid(),

      nn.Linear(32, 16),
      nn.Sigmoid(),

      nn.Linear(16, 32),
      nn.Sigmoid(),

      nn.Linear(32, 2),
      
      nn.Softmax(dim=0)
    )

  def forward(self, x):
    logits = self.linear_stack(x)

    return logits