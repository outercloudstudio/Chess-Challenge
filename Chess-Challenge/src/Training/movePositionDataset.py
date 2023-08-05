from torch.utils.data import Dataset

class MovePositionDataset(Dataset):
    def __init__(
        self,
        dataset_file,
        transform,
        targetTransform
    ):
        file = open(dataset_file, "r")

        entries = file.read().split("\n")

        file.close()

        self.fens = []
        self.moves = []
        self.evaluations = []

        for entry in entries:
          if entry == "":
            continue
          
          items = entry.split(" | ")

          self.fens.append(items[0])
          self.moves.append(items[1])
          self.evaluations.append(float(items[2]))
             
        self.transform = transform
        self.targetTransform = targetTransform

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        uci = self.moves[idx]
        evaluation = self.evaluations[idx]

        return self.transform(fen, uci), self.targetTransform(evaluation)
