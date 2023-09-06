import chess
import torch
import os
import random

from torch import nn
from model import LilaEvaluationModel
from modelConverter import convert

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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

modelName = "Lila_1"

if os.path.exists(
  "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
):
  model.load_state_dict(
    torch.load(
      "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
    )
  )

model.train()

decisions = []

def boardToTensor(board):
  boardTensor = torch.zeros(8, 8, dtype=torch.float32)

  for x in range(8):
    for y in range(8):
      if board.piece_at(chess.square(x, y)) != None:
        boardTensor[x, y] = board.piece_at(chess.square(x, y)).piece_type * (1 if board.piece_at(chess.square(x, y)).color == chess.WHITE else -1)


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

  decisions.append(decision)

  return move

def simulateGame():
  print("Simulating game...")

  board = chess.Board(fens[random.randint(0, len(fens) - 1)])

  while board.outcome() == None:
    decision = makeDecision(board)

    board.push(decision)

    print('\n')
    print(board)

  if board.outcome().winner == chess.WHITE:
    return 1
  
  if board.outcome().winner == chess.BLACK:
    return -1
  
  return 0

def initializeTraingingState():
  print("Initializing training state...")

  global decisions

  decisions = []

def train(outcome):
  print("Training...")

  predictions = torch.tensor(decisions, dtype=torch.float32).to(device)
  outcomes = torch.full(predictions.size(), outcome, dtype=torch.float32).to(device)

  print(predictions)
  print(outcomes)

  loss = loss_fn(predictions, outcomes)

  print(f"Loss: {loss}")

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

def saveModel():
  print("Saving model...")

  torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Training\\Models\\{modelName}.pth")

  convert(modelName)

while True:
  initializeTraingingState()

  outcome = simulateGame()

  train(outcome)

  saveModel()