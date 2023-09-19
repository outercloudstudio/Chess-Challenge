import chess
import torch
import os
import random
import time

from model import LilaModel

from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

model = LilaModel().to(device)

modelName = "Lila_8"

if os.path.exists(
  "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
):
  model.load_state_dict(
    torch.load(
      "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
    )
  )

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

def makeDecision(board):
  legalMoves = list(board.legal_moves)
  predictions = []

  for move in legalMoves:
    board.push(move)

    prediction = model(boardToTensor(board).to(device))

    predictions.append(prediction)

    board.pop()

  decision = None
  move = None

  for i in range(len(predictions)):
    if board.turn == chess.WHITE:
      if decision == None or predictions[i].item() > decision.item():
        decision = predictions[i]
        move = legalMoves[i]
    else:
      if decision == None or predictions[i].item() < decision.item():
        decision = predictions[i]
        move = legalMoves[i]

  return move, decision

def simulateGame():
  board = chess.Board()

  while board.outcome() == None:
    move, prediction = makeDecision(board)

    board.push(move)

    print(board)
    print(prediction.item() * 3000)

    input("Press enter to continue...")

model.eval()

# while True:
#   simulateGame()

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = []

file = open("D:/Chess-Challenge/Training/Models/Lila_8.txt", "r")

index = 0

for line in file.readlines():
  if(line == "\n"): break

  data.append(float(line))

  index += 1

sns.kdeplot(data, bw_method=0.25)
plt.show()