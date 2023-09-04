import torch

from model import EvaluationNeuralNetwork

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def convert(name):
  model = EvaluationNeuralNetwork().to(device)

  model.load_state_dict(
    torch.load(
      "D:\\Chess-Challenge\\Training\\Models\\" + name + ".pth"
    )
  )

  weightCount = 0

  weightOutput = ""

  for param in model.parameters():
    weightCount += param.data.flatten().size()[0]

    for value in param.data.flatten().tolist():
      weightOutput += str(value) + "\n"

  print(f"Converted {weightCount} weights. Compressed size: {weightCount / 8}")

  weightFile = open(
    f"D:\\Chess-Challenge\\Training\\Models\\{name}.txt", "w"
  )
  weightFile.write(weightOutput)
  weightFile.close()