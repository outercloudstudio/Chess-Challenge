from torch.utils.data import Dataset

class ChessDataset(Dataset):
  def __init__(self, transform=None, target_transform=None):
    self.positions = []
    self.evaluation = []

    file = open("D:/Chess-Challenge/Training/Data/sebv2.epd", "r")
    
    for line in file.readlines():
      items = line.split(" | ")

      if len(items) < 3: continue

      self.positions.append(items[0])
      self.evaluation.append(float(items[1]) / 1000)

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.positions)

  def __getitem__(self, idx):
    fen = self.positions[idx]
    evaluation = self.evaluation[idx]

    if self.transform:
      fen = self.transform(fen)
    if self.target_transform:
      evaluation = self.target_transform(evaluation)

    return fen, evaluation