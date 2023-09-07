import chess
import torch
import os
import random

from model import LilaEvaluationModel

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

model = LilaEvaluationModel().to(device)

modelName = "Lila_2"

if os.path.exists(
  "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
):
  model.load_state_dict(
    torch.load(
      "D:\\Chess-Challenge\\Training\\Models\\" + modelName + ".pth"
    )
  )

model.eval()

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
  board = chess.Board()

  botMove = random.randint(0, 1) == 0

  while board.outcome() == None:
    if botMove:
      move, prediction = makeDecision(board)

      board.push(move)
    else:
      move = input("Enter Move")

      board.push_uci(move)

    print(board)

    botMove = not botMove

while True:
  simulateGame()