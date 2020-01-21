
def collate(batch):
  data = []
  label = []
  nameFile = []
  for dp in batch:
    data.append(dp[0])
    label.append(dp[1])
    nameFile.append(dp[2])
  return (data, label, nameFile)