from torch import nn

class LilaEvaluationModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.linear_stack = nn.Sequential(
      nn.Linear(16, 16),
      nn.Tanh(),

      nn.Linear(16, 16),
      nn.Tanh(),

      nn.Linear(16, 16),
      nn.Tanh(),

      nn.Linear(16, 9),
    )

  def forward(self, x):
    logits = self.linear_stack(x)

    return logits