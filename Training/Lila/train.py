import chess
import torch
import os
import random
import time
import math

from torch import nn
from model import LilaModel
from modelConverter import convert
from stockfish import Stockfish

stockfish = Stockfish(path="D:/Chess-Challenge/Training/Stockfish16.exe", depth=5, parameters={ "Threads": 4, "Hash": 1024 })

fensFile = open('D:/Chess-Challenge/Training/Data/Fens.txt', 'r')
fens = fensFile.readlines()
fensFile.close()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

model = LilaModel().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

modelName = "Lila_5"

if os.path.exists(
  "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
):
  model.load_state_dict(
    torch.load(
      "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
    )
  )

model.train()

def boardToTensor(board):
  boardTensor = torch.zeros(6 * 64, dtype=torch.float32)

  for x in range(8):
    for y in range(8):
      if board.piece_at(chess.square(x, y)) != None:
        piece = board.piece_at(chess.square(x, y))

        if piece == None: continue

        boardTensor[x * 8 * 6 + y * 6 + piece.piece_type - 1] =  1 if piece.color == chess.WHITE else -1

  return boardTensor

def saveModel():
  print("Saving model...")

  torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Training\\Models\\{modelName}.pth")

  convert(modelName)

  
position = 0

while True:
  fen = fens[random.randint(0, len(fens) - 1)]
  board = chess.Board(fen)

  prediction = model(boardToTensor(board).to(device))

  stockfish.set_fen_position(fen)
  stockfishEvaluationData = stockfish.get_evaluation()
  stockfishEvaluation = stockfishEvaluationData["value"] / 3000
  if stockfishEvaluation > 1: stockfishEvaluation = 1
  if stockfishEvaluation < -1: stockfishEvaluation = -1

  if stockfishEvaluationData["type"] == "mate" and stockfishEvaluationData["value"] != 0:
    stockfishEvaluation = abs(stockfishEvaluationData["value"]) / stockfishEvaluationData["value"]

  target = torch.tensor([stockfishEvaluation], dtype=torch.float32).to(device)

  loss = loss_fn(prediction, target)

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  position += 1

  print(f"Loss: {round(loss.item(), 3)} Prediction {round(prediction.item(), 3)} Target: {round(target.item(), 3)} Position: {position}")


  if position % 100 == 0: saveModel()