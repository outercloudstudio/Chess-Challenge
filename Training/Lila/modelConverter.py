import torch

from model import LilaEvaluationModel

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def convert(name):
  model = LilaEvaluationModel().to(device)

  model.load_state_dict(
    torch.load(
      "D:\\Chess-Challenge\\Training\\Models\\" + name + ".pth"
    )
  )

  weightCount = 0

  weightOutput = ""

  pruned = 0

  for param in model.parameters():
    weightCount += param.data.flatten().size()[0]

    for value in param.data.flatten().tolist():
      if abs(value) < 0.04:
        pruned += 1

      weightOutput += str(value) + "\n"

  

  print(f"Converted {weightCount} weights. Compressed size: {(weightCount - pruned / 2.0) / 17.3}")

  weightFile = open(
    f"D:\\Chess-Challenge\\Training\\Models\\{name}.txt", "w"
  )
  weightFile.write(weightOutput)
  weightFile.close()