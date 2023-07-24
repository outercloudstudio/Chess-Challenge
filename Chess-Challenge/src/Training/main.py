import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import EvaluationNeuralNetwork
from dataset import PositionDataset, positionToTensor

training_data = PositionDataset('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Evaluations Medium.txt')
test_data = PositionDataset('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Evaluations Small.txt')

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

model = EvaluationNeuralNetwork().to(device)

# model.load_state_dict(torch.load("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 3.pth"))

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)

    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
  num_batches = len(dataloader)

  model.eval()

  test_loss = 0

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)

      pred = model(X)

      test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

torch.save(model.state_dict(), "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 3.pth")

weightCount = 0

weightOutput = ''

for param in model.parameters():
  weightCount += param.data.flatten().size()[0]

  for value in param.data.flatten().tolist():
    weightOutput += str(value) + '\n'

print(weightCount, weightCount / 8)

weightFile = open("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 3.txt", 'w')
weightFile.write(weightOutput)
weightFile.close()

# model.eval()

# with torch.no_grad():
#   inputTensor = positionToTensor('r4rk1/pp1nqppp/2p1p1b1/3pPn2/1P1P4/2P2N2/P2NBPPP/R2Q1RK1 w - - 1 12').unsqueeze(0)

#   print(model(inputTensor.to(device)))