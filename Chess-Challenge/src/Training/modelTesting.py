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

model.eval()

with torch.no_grad():
  print(training_data.fens[29280])

  X, y = training_data[29280]
  X = X.to(device)
  y = y.to(device)

  pred = model(X)

  loss = loss_fn(pred, y).item()

  print(f"prediction: {pred.item()} loss: {loss:>8f} \n")

  # for X, y in dataloader:
  #   X, y = X.to(device), y.to(device)

  #   pred = model(X)

  #   test_loss += loss_fn(pred, y).item()

  # test_loss /= num_batches

  # print(f"Test Error: Avg loss: {test_loss:>8f} \n")