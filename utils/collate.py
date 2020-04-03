import torch 

def collate(batch):
  data = []
  label = []
  nameFile = []
  for dp in batch:
    data.append(torch.tensor(dp[0]))
    label.append(torch.tensor(dp[1]))
    nameFile.append((dp[2]))
  return (data, label, nameFile)

