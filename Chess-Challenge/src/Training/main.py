import os.path
import chess
from stockfish import Stockfish
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import EvaluationNeuralNetwork
from dataset import PositionDataset, positionToTensor

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

modelName = "ARCNET 4"

if os.path.exists("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"): model.load_state_dict(torch.load("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"))

stockfish = Stockfish(path="D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\eval.exe", depth=10, parameters={"Threads": 2, "Minimum Thinking Time": 0, "Hash": 2048})

def evaluatePosition(board):
  fen = board.fen()

  stockfish.set_fen_position(fen)

  evaluation = stockfish.get_evaluation()

  if evaluation['type'] == 'cp': return evaluation['value'] / 100
  if evaluation['type'] == 'mate': return evaluation['value'] / 10


def playTraingingGame():
  model.train()

  board = chess.Board()

  while(board.outcome() == None):
    moves = list(board.legal_moves)

    # Pick Move
    evaluations = []
    actualEvaluations = []

    averageLoss = 0

    for move in moves:
      board.push(move)

      prediction = model(positionToTensor(board.fen()).to(device))
      predictedEvaluation = prediction.item()
      actualEvaluation = evaluatePosition(board)

      evaluations.append(predictedEvaluation)
      actualEvaluations.append(actualEvaluation)

      board.pop()

      # Compute prediction error
      loss = loss_fn(prediction, torch.tensor([actualEvaluation], dtype=torch.float32).to(device))

      # Backpropagation
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      averageLoss += loss
    
    bestEvaluation = max(evaluations)
    if board.turn == chess.BLACK: bestEvaluation = min(evaluations)

    bestMove = moves[evaluations.index(bestEvaluation)]
    bestActualEvaluation = actualEvaluations[evaluations.index(bestEvaluation)]
    board.push(bestMove)

    averageLoss /= len(moves)

    print(board)
    print(f"Average loss: {averageLoss:>7f}")
    print(f"Predicted evaluation: {bestEvaluation} Actual evaluation: {bestActualEvaluation}")

  torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\{modelName}.pth")

while True:
  playTraingingGame()

# training_data = PositionDataset('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Evaluations Medium.txt')
# test_data = PositionDataset('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Evaluations Small.txt')

# batch_size = 32

# train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# print(f"Using {device} device")

# model = EvaluationNeuralNetwork().to(device)

# # model.load_state_dict(torch.load("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 3.pth"))

# loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# def train(dataloader, model, loss_fn, optimizer):
#   size = len(dataloader.dataset)
#   model.train()
#   for batch, (X, y) in enumerate(dataloader):
#     X, y = X.to(device), y.to(device)

#     # Compute prediction error
#     pred = model(X)

#     loss = loss_fn(pred, y)

#     # Backpropagation
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#     if batch % 100 == 0:
#       loss, current = loss.item(), (batch + 1) * len(X)
#       print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def test(dataloader, model, loss_fn):
#   num_batches = len(dataloader)

#   model.eval()

#   test_loss = 0

#   with torch.no_grad():
#     for X, y in dataloader:
#       X, y = X.to(device), y.to(device)

#       pred = model(X)

#       test_loss += loss_fn(pred, y).item()

#     test_loss /= num_batches

#     print(f"Test Error: Avg loss: {test_loss:>8f} \n")

# epochs = 1
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)

# torch.save(model.state_dict(), "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 3.pth")

# # model.eval()

# # with torch.no_grad():
# #   inputTensor = positionToTensor('r4rk1/pp1nqppp/2p1p1b1/3pPn2/1P1P4/2P2N2/P2NBPPP/R2Q1RK1 w - - 1 12').unsqueeze(0)

# #   print(model(inputTensor.to(device)))