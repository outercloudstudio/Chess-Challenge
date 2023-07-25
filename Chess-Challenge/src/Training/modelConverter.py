import chess
import torch
from torch import nn
from dataset import positionToTensor

from model import EvaluationNeuralNetwork

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = EvaluationNeuralNetwork().to(device)

modelName = "ARCNET 4"

model.load_state_dict(
    torch.load(
        "D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\" + modelName + ".pth"
    )
)

model.eval()

with torch.no_grad():
    board = chess.Board()

    while board.outcome() == None:
        if board.turn == chess.WHITE:
            print(board)

            # move = input("Move > ")

            # board.push_uci(move)

            # continue

        moves = list(board.legal_moves)

        evaluations = []

        for move in moves:
            board.push(move)

            prediction = model(positionToTensor(board.fen()).to(device))
            predictedEvaluation = prediction.item()

            evaluations.append(predictedEvaluation)

            board.pop()

        bestEvaluation = max(evaluations)
        if board.turn == chess.BLACK:
            bestEvaluation = min(evaluations)

        bestMove = moves[evaluations.index(bestEvaluation)]
        board.push(bestMove)

    print(board)
    print(board.outcome())

weightCount = 0

weightOutput = ""

for param in model.parameters():
    weightCount += param.data.flatten().size()[0]

    for value in param.data.flatten().tolist():
        weightOutput += str(value) + "\n"

print(weightCount, weightCount / 8)

weightFile = open(
    f"D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\{modelName}.txt", "w"
)
weightFile.write(weightOutput)
weightFile.close()
