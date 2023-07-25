import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 8080))

print('Connected!')

message_to_send = "Hello, server!"
client.send(message_to_send.encode('utf-8'))

while True:
  data = client.recv(1024)

  message = data.decode('utf-8')

  print("Received from server:", message)

# import os.path
# import chess
# import random
# from stockfish import Stockfish
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# from model import EvaluationNeuralNetwork
# from dataset import positionToTensor
# from modelConverter import convert

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

# model = EvaluationNeuralNetwork().to(device)

# loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# modelName = "ARCNET 5"

# if os.path.exists(
#     "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"
# ):
#     model.load_state_dict(
#         torch.load(
#             "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"
#         )
#     )

# stockfish = Stockfish(
#     path="D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\eval.exe",
#     depth=10,
#     parameters={"Threads": 2, "Minimum Thinking Time": 0, "Hash": 2048},
# )


# def evaluatePosition(board):
#     fen = board.fen()

#     stockfish.set_fen_position(fen)

#     evaluation = stockfish.get_evaluation()

#     if evaluation["type"] == "cp":
#         return evaluation["value"] / 100
#     if evaluation["type"] == "mate":
#         return evaluation["value"] * 10

# def train():
#   model.train()

#   completed = 0

#   games = []

#   for i in range(40):
#     games.append(chess.Board())

#   while True:
#     for gameIndex in range(len(games)):
#       board = games[gameIndex]

#       if board.outcome() != None:
#         completed += 1

#         games[gameIndex] = chess.Board()
        
#         torch.save(model.state_dict(), f"D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\{modelName}.pth")

#         convert(modelName)

#         continue

#       moves = list(board.legal_moves)
#       moveChoices = []

#       for move in moves:
#         board.push(move)

#         modelEvaluation = model(positionToTensor(board.fen()).to(device)).item()
        
#         moveChoices.append({
#           "move": move,
#           "evaluation": modelEvaluation
#         })

#         board.pop()

#       moveChoices.sort(key=lambda x: x["evaluation"], reverse=board.turn == chess.WHITE)

#       choice = moveChoices[random.randint(0, min(2, len(moveChoices) - 1))]["move"]
      
#       board.push(choice)

#       prediction = model(positionToTensor(board.fen()).to(device))
#       actual = evaluatePosition(board)

#       loss = loss_fn(
#         prediction,
#         torch.tensor([actual], dtype=torch.float32).to(device),
#       )

#       loss.backward()
#       optimizer.step()
#       optimizer.zero_grad()

#       print(chr(27) + "[2J")
#       print(board)
#       print(f"Predicted evaluation: {prediction.item()} Actual evaluation: {actual} Loss: {loss}")
#       print(f"Completed {completed} games")
          

# train()

# # def playTraingingGame():
# #     model.train()

# #     board = chess.Board()

# #     while board.outcome() == None:
# #         moves = list(board.legal_moves)

# #         evaluations = []
# #         actualEvaluations = []

# #         averageLoss = 0

# #         for move in moves:
# #             board.push(move)

# #             prediction = model(positionToTensor(board.fen()).to(device))
# #             predictedEvaluation = prediction.item()
# #             actualEvaluation = evaluatePosition(board)

# #             evaluations.append(predictedEvaluation)
# #             actualEvaluations.append(actualEvaluation)

# #             board.pop()

# #             loss = loss_fn(
# #                 prediction,
# #                 torch.tensor([actualEvaluation], dtype=torch.float32).to(device),
# #             )

# #             loss.backward()
# #             optimizer.step()
# #             optimizer.zero_grad()

# #             averageLoss += loss

# #         bestEvaluation = max(evaluations)
# #         if board.turn == chess.BLACK:
# #             bestEvaluation = min(evaluations)

# #         bestMove = moves[evaluations.index(bestEvaluation)]
# #         bestActualEvaluation = actualEvaluations[evaluations.index(bestEvaluation)]
# #         board.push(bestMove)

# #         averageLoss /= len(moves)

# #         print(board)
# #         print(f"Average loss: {averageLoss:>7f}")
# #         print(
# #             f"Predicted evaluation: {bestEvaluation} Actual evaluation: {bestActualEvaluation}"
# #         )

# #     torch.save(
# #         model.state_dict(),
# #         f"D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\{modelName}.pth",
# #     )

# #     convert(modelName)
