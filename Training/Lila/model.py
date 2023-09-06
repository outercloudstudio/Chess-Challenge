from torch import nn

class LilaEvaluationModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.linearStack = nn.Sequential(
      nn.Conv2d(1, 32, 3, padding=1),
      nn.Tanh(),

      nn.Conv2d(32, 32, 3, padding=1),
      nn.Tanh(),

      nn.Conv2d(32, 1, 3, padding=1),
      nn.Tanh(),
      
      nn.AvgPool2d(8)
    )

  def forward(self, x):
    return self.linearStack(x)