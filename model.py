import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN(nn.Module):
  def __init__(self, num_classes, n_frames, n_shift, device):
    super(CNN_RNN, self).__init__()
    self.device = device
    self.n_frames = n_frames
    self.n_shift = n_shift
    self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=2)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2)
    self.gru = nn.GRUCell(input_size=1920, hidden_size=1920)
    self.fc2 = nn.Linear(1920, num_classes)
  
  def forward(self, x, seq_length):
    #deep_features = []
    #for signal in signals:
    #  for window_context in 
    #    x = signal[:, ]
    x = F.max_pool2d(F.sigmoid(self.conv1(x)), kernel_size=3, stride=3, padding=0)
    x = F.max_pool2d(F.sigmoid(self.conv2(x)), kernel_size=3, stride=3, padding=0)
    x = x.view(-1, 1, self.num_flat_features(x))
    hx = torch.randn(1, 1920).to(self.device)
    for i in range(seq_length):
      hx = self.gru(x[i], hx)
    x = self.fc2(hx)
    return x
  
  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features