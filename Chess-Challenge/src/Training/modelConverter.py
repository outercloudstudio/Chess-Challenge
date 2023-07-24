import torch
from torch import nn
from dataset import positionToTensor

from model import EvaluationNeuralNetwork

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = EvaluationNeuralNetwork().to(device)

modelName = "ARCNET 4"

model.load_state_dict(
    torch.load(
        "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"
    )
)

fen = "rnbq3r/pp2pkbp/2pp1np1/8/3PP3/5N2/PPP2PPP/RNBQK2R w KQ - 0 7"

model.eval()

with torch.no_grad():
    print(model(positionToTensor(fen).to(device)).item())

weightCount = 0

weightOutput = ""

# for name, param in model.named_parameters():
#     print(name)

# print(list(model.parameters())[1])
# print(list(model.parameters())[1].shape)

convt1 = nn.Conv2d(1, 16, 3, 1, 1)
convt1.weight = list(model.parameters())[0]
convt1.bias = list(model.parameters())[1]

res = convt1(positionToTensor(fen).to(device))

print(res)
print(res.shape)

# print(convt1.shape)
# convt2 = nn.Conv2d(16, 8, 3, 1, 1)(convt1)
# print(convt2.shape)

for param in model.parameters():
    weightCount += param.data.flatten().size()[0]

    for value in param.data.flatten().tolist():
        weightOutput += str(value) + "\n"

print(weightCount, weightCount / 8)

weightFile = open(
    f"D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\{modelName}.txt", "w"
)
weightFile.write(weightOutput)
weightFile.close()
