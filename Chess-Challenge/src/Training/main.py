import socket
import chess
import torch
import time
import os
import random

from torch import nn
from model import EvaluationNeuralNetwork
from modelConverter import convert

fensFile = open('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Fens\\FensLarge.txt', 'r')
fens = fensFile.readlines()
fensFile.close()

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 8080))

print('Connected!')

time.sleep(1)

def sendString(string):
  client.send(len(string).to_bytes(4, 'little'))
  client.send(string.encode('utf-8'))

def readInt():
  return int.from_bytes(client.recv(4), 'little')

def readBool():
  return bool.from_bytes(client.recv(1), 'little')

def getState(board, move):
  state = []

  sendString(board.fen())
  sendString(move.uci())

  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
  state.append(1 if readBool() else 0)
  state.append(1 if readBool() else 0)
  state.append(1 if readBool() else 0)
  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
  state.append(readInt())
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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

modelName = "ARCNET 7"

if os.path.exists(
  "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"
):
  model.load_state_dict(
    torch.load(
        "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"
    )
  )

model.train()

board = chess.Board()
predictions = []

def staticEvaluation(board):
  if board.is_checkmate():
    return 100 if board.turn == chess.WHITE else -100
  
  if board.outcome() != None and board.outcome().winner == None:
    return 0

  return 0

def searchMove(board, move, depth):
  if depth == 0 or board.outcome() != None: return staticEvaluation(board)
  
  bestMoveEvaluation = None

  for move in board.legal_moves:
    board.push(move)

    evaluation = searchMove(board, move, depth - 1)

    if bestMoveEvaluation == None:
      bestMoveEvaluation = evaluation
    elif board.turn == chess.WHITE:
      bestMoveEvaluation = max(bestMoveEvaluation, evaluation)
    else:
      bestMoveEvaluation = min(bestMoveEvaluation, evaluation)

    board.pop()
  
  return bestMoveEvaluation

while True:
  if(board.outcome() != None):
    board = chess.Board(fens[random.randint(0, len(fens) - 1)])
    predictions = []

    continue

  moves = list(board.legal_moves)
  moveChoices = []

  # if(len(board.piece_map()) <= 4):
  #   for move in moves:
  #     evaluation = searchMove(board, move, 3)

  #     whiteWinPercentage = (evaluation + 100) / 200
  #     blackWinPercentage = 1 - whiteWinPercentage

  #     moveChoices.append({
  #       "move": move,
  #       "evaluation": torch.tensor([whiteWinPercentage, blackWinPercentage], dtype=torch.float32)
  #     })
  # else:
  for move in moves:
    board.push(move)

    if(board.is_repetition(2)):
      board.pop()

      moveChoices.append({
        "move": move,
        "evaluation": torch.tensor([0, 1], dtype=torch.float32) if board.turn == chess.WHITE else torch.tensor([1, 0], dtype=torch.float32)
      })

      continue

    if(board.is_variant_draw()):
      board.pop()

      moveChoices.append({
        "move": move,
        "evaluation": torch.tensor([0, 1], dtype=torch.float32) if board.turn == chess.WHITE else torch.tensor([1, 0], dtype=torch.float32)
      })

      continue

    if board.is_checkmate():
      board.pop()

      moveChoices.append({
        "move": move,
        "evaluation": torch.tensor([1, 0], dtype=torch.float32) if board.turn == chess.WHITE else torch.tensor([0, 1], dtype=torch.float32)
      })

      continue

    board.pop()

    state = getState(board, move).to(device)

    modelEvaluation = model(state)

    moveChoices.append({
      "move": move,
      "evaluation": modelEvaluation
    })

  moveChoices.sort(key=lambda x: x["evaluation"][0], reverse=board.turn == chess.WHITE)

  choice = moveChoices[0]["move"]

  state = getState(board, choice).to(device)

  prediction = model(state).to(device)

  predictions.append(prediction)
  
  board.push(choice)

  if board.outcome() == None: continue

  averageLoss = 0

  actual = torch.tensor([0, 0], dtype=torch.float32)

  if(board.outcome().winner == chess.WHITE):
    actual[0] = 1

  if(board.outcome().winner == chess.BLACK):
    actual[1] = 1

  if(board.outcome().winner == None):
    actual[0] = random.randrange(0, 100) / 100
    actual[1] = random.randrange(0, 100) / 100
  
  actual = actual.to(device)

  for prediction in predictions:
    loss = loss_fn(prediction, actual)

    averageLoss += loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  averageLoss /= len(predictions)

  print(f"Loss: {averageLoss}")

  torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\{modelName}.pth")

  convert(modelName)

  predictions = []

  # board = chess.Board(fens[random.randint(0, len(fens) - 1)])
  board = chess.Board()