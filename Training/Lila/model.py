from torch import nn

class LilaEvaluationModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.linearStack = nn.Sequential(
      nn.Conv2d(1, 24, 3, padding=1),
      nn.Tanh(),

      nn.Conv2d(24, 24, 3, padding=1),
      nn.Tanh(),

      nn.Conv2d(24, 24, 3, padding=1),
      nn.Tanh(),

      nn.Conv2d(24, 24, 3, padding=1),
      nn.Tanh(),

      nn.Conv2d(24, 1, 3, padding=1),
      nn.Tanh(),
      
      nn.AvgPool2d(8)
    )

  def forward(self, x):
    return self.linearStack(x)