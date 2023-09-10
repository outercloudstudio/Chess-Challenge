import chess
import torch
import os
import random
import time
import math

from torch import nn
from model import LilaEvaluationModel
from modelConverter import convert
from stockfish import Stockfish

stockfish = Stockfish(path="D:/Chess-Challenge/Training/Stockfish16.exe", depth=5, parameters={ "Threads": 4, "Hash": 1024 })

fensFile = open('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Fens\\FensLarge.txt', 'r')
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

model = LilaEvaluationModel().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

modelName = "Lila_3"

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
  boardTensor = torch.zeros(1, 8, 8, dtype=torch.float32)

  for x in range(8):
    for y in range(8):
      if board.piece_at(chess.square(x, y)) != None:
        boardTensor[0, x, y] = board.piece_at(chess.square(x, y)).piece_type * (1 if board.piece_at(chess.square(x, y)).color == chess.WHITE else -1)

  return boardTensor

def makeDecision(board):
  legalMoves = list(board.legal_moves)
  predictions = []

  for move in legalMoves:
    board.push(move)

    prediction = model(boardToTensor(board).to(device))

    predictions.append(prediction)

    board.pop()

  pairings = []

  for i in range(len(predictions)):
    pairings.append((predictions[i], legalMoves[i]))

  pairings.sort(key=lambda x: x[0].item(), reverse=board.turn == chess.WHITE)

  decision, move = pairings[min(math.floor(pow(random.random(), 3) * 3), len(pairings) - 1)]

  return move, decision

def simulateGame():
  board = chess.Board(fens[random.randint(0, len(fens) - 1)])

  while board.outcome() == None:
    move, prediction = makeDecision(board)

    board.push(move)

    train(prediction, board)

def train(prediction, board):
  stockfish.set_fen_position(board.fen())
  stockfishEvaluationData = stockfish.get_evaluation()
  stockfishEvaluation = stockfishEvaluationData["value"] / 3000
  if stockfishEvaluation > 1: stockfishEvaluation = 1
  if stockfishEvaluation < -1: stockfishEvaluation = -1

  if stockfishEvaluationData["type"] == "mate" and stockfishEvaluationData["value"] != 0:
    stockfishEvaluation = abs(stockfishEvaluationData["value"]) / stockfishEvaluationData["value"]

  target = torch.tensor([[[stockfishEvaluation]]], dtype=torch.float32).to(device)

  loss = loss_fn(prediction, target)

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  print(f"Loss: {round(loss.item(), 3)} Prediction {round(prediction.item(), 3)} Target: {round(target.item(), 3)})")


def saveModel():
  print("Saving model...")

  torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Training\\Models\\{modelName}.pth")

  convert(modelName)

while True:
  simulateGame()

  saveModel()