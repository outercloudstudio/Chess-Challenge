from torch.utils.data import Dataset

class ChessDataset(Dataset):
  def __init__(self, transform=None, target_transform=None):
    self.positions = []
    self.winPercent = []

    file = open("D:/Chess-Challenge/Training/Data/sebv2.epd", "r")
    
    for line in file.readlines():
      items = line.split(" | ")

      if len(items) < 3: continue

      self.positions.append(items[0])
      self.winPercent.append(float(items[2]) * 2 - 1)

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.positions)

  def __getitem__(self, idx):
    fen = self.positions[idx]
    winPercent = self.winPercent[idx]

    if self.transform:
      fen = self.transform(fen)
    if self.target_transform:
      winPercent = self.target_transform(winPercent)

    return fen, winPercent