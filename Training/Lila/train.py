import chess
import torch
import os
import random
import time

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
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

modelName = "Lila_2"

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
  board = chess.Board(fens[random.randint(0, len(fens) - 1)])

  while board.outcome() == None:
    move, prediction = makeDecision(board)

    board.push(move)

    train(prediction, board)

def train(prediction, board):
  stockfish.set_fen_position(board.fen())
  stockfishEvaluationData = stockfish.get_evaluation()
  stockfishEvaluation = stockfishEvaluationData["value"] / 1000

  if stockfishEvaluationData["type"] == "mate" and stockfishEvaluationData["value"] != 0:
    abs(stockfishEvaluationData["value"]) / stockfishEvaluationData["value"]

  target = torch.tensor([[[stockfishEvaluation]]], dtype=torch.float32).to(device)

  loss = loss_fn(prediction, target)

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  print(loss.item())


def saveModel():
  print("Saving model...")

  torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Training\\Models\\{modelName}.pth")

  convert(modelName)

while True:
  simulateGame()

  saveModel()