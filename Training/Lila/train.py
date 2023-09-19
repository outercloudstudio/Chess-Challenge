import chess
import torch
import os
import random

from torch import nn
from torch.utils.data import DataLoader
from model import LilaModel
from modelConverter import convert
from data import ChessDataset


device = (
    "cuda"
    if torch.cuda.is_available  ()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

model = LilaModel().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

modelName = "Lila_8"

if os.path.exists(
  "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
):
  model.load_state_dict(
    torch.load(
      "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
    )
  )

model.train()

def positionToTensor(position):
  return boardToTensor(chess.Board(position))

def boardToTensor(board):
  boardTensor = torch.zeros(6 * 64 + 1, dtype=torch.float32)

  for x in range(8):
    for y in range(8):
      if board.piece_at(chess.square(x, y)) != None:
        piece = board.piece_at(chess.square(x, y))

        if piece == None: continue

        boardTensor[x * 8 * 6 + y * 6 + piece.piece_type - 1] =  1 if piece.color == chess.WHITE else -1

  boardTensor[384] = 1 if board.turn == chess.WHITE else -1

  return boardTensor

def saveModel():
  print("Saving model...")

  torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Training\\Models\\{modelName}.pth")

  convert(modelName)

data = ChessDataset(transform = positionToTensor, target_transform = torch.tensor)
dataLoader = DataLoader(data, batch_size=1, shuffle=True)

position = 0
positionsSinceLoss = 0
averageLoss = 0

averageLosses = []

for positionTensor, evaluationTensor in dataLoader:
  prediction = model(positionTensor.to(device))

  loss = loss_fn(prediction, evaluationTensor.to(device))

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  averageLoss += loss.item()
  positionsSinceLoss += 1
  position += 1

  # print(f"Prediction: {round(prediction.item(), 3)} Actual: {round(winPercentTensor.item(), 3)}")

  if position % 100 == 0: 
    averageLosses.append(averageLoss / positionsSinceLoss)

    open("./history.txt", "w").write("\n".join([str(x) for x in averageLosses]))

    print(f"Loss: {round(averageLoss / positionsSinceLoss, 3)} Positions Trained: {position}")

    positionsSinceLoss = 0
    averageLoss = 0

    saveModel()