import socket
import chess
import torch
import time
import os
import random
import struct

from torch import nn
from torch.utils.data import DataLoader
from model import EvaluationNeuralNetwork
from modelConverter import convert
from dataset import PositionDataset, positionToTensor
# from stockfish import Stockfish

# stockfish = Stockfish(
#   path="D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\eval.exe",
#   depth=10,
#   parameters={"Threads": 2, "Minimum Thinking Time": 0, "Hash": 2048},
# )

# def evaluatePosition(board):
#   fen = board.fen()
#   stockfish.set_fen_position(fen)
#   evaluation = stockfish.get_evaluation()
#   if evaluation["type"] == "cp":
#       return evaluation["value"] / 100
#   if evaluation["type"] == "mate":
#       return evaluation["value"] * 10

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 8080))

print('Connected!')

def sendString(string):
  client.send(len(string).to_bytes(4, 'little'))
  client.send(string.encode('utf-8'))

def readFloat():
  return struct.unpack('<f', client.recv(4))[0]

def readInt():
  return int.from_bytes(client.recv(4), 'little')

def readBool():
  return bool.from_bytes(client.recv(1), 'little')

def getState(fen):
  state = []

  sendString(fen)

  state.append(readFloat())
  state.append(readFloat())
  state.append(readInt())

  return torch.tensor(state, dtype=torch.float32)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = EvaluationNeuralNetwork().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

modelName = "ARCNET 2"

if os.path.exists(
  "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"
):
  model.load_state_dict(
    torch.load(
        "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"
    )
  )

training_data = PositionDataset('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Evaluations Medium.txt', getState)
test_data = PositionDataset('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Evaluations Small.txt', getState)

batch_size = 4

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    pred = model(X)

    loss = loss_fn(pred, y)

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

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\{modelName}.pth")
convert(modelName)