import torch

from model import LilaModel

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def convert(name):
  model = LilaModel().to(device)

  model.load_state_dict(
    torch.load(
      "D:\\Chess-Challenge\\Training\\Models\\" + name + ".pth"
    )
  )

  paramCount = 0

  paramOutput = ""

  for param in model.parameters():
    paramCount += param.data.flatten().size()[0]

  print(f"Converted {paramCount} weights. Compressed size: {paramCount / 16}")

  weightFile = open(
    f"D:\\Chess-Challenge\\Training\\Models\\{name}.txt", "w"
  )
  weightFile.write(paramOutput)
  weightFile.close()