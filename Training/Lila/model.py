from torch import nn

class LilaEvaluationModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.linearStack = nn.Sequential(
      nn.Linear(64, 16),
      nn.Tanh(),

      nn.Linear(16, 16),
      nn.Tanh(),

      nn.Linear(16, 16),
      nn.Tanh(),

      nn.Linear(16, 1),
      nn.Tanh()
    )

  def forward(self, x):
    return self.linearStack(x)