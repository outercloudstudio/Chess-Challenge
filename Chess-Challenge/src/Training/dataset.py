import torch
from torch.utils.data import Dataset
import chess


def getFen(entry):
    return entry.split(" | ")[0]


def getEvaluation(entry):
  continousEvaluation = float(entry.split(" | ")[1])

  if continousEvaluation > 0.2:
    return 1
  elif continousEvaluation < 0.2:
    return -1
  else:
    return 0


def positionToTensor(fen):
    board = chess.Board(fen)
    tensor = torch.zeros((1, 8, 8), dtype=torch.float32)

    for x in range(8):
        for y in range(8):
            piece = board.piece_at(square=chess.square(x, y))

            if piece == None:
                tensor[0][7 - y][7 - x] = 0
            else:
                tensor[0][7 - y][7 - x] = piece.piece_type * (
                    1 if piece.color == chess.WHITE else -1
                )

    return tensor


def evalutionToTensor(evaluation):
    return torch.tensor([evaluation], dtype=torch.float32)


class PositionDataset(Dataset):
    def __init__(
        self,
        dataset_file,
        transform
    ):
        file = open(dataset_file, "r")

        entries = file.read().split("\n")

        self.fens = list(map(getFen, entries))
        self.evaluation = list(map(getEvaluation, entries))
        file.close()
        
        self.transform = transform

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.transform(self.fens[idx])
        evaluation = evalutionToTensor(self.evaluation[idx])

        return fen, evaluation
