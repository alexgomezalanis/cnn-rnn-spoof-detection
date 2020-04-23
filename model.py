import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class CNN_RNN(nn.Module):
  def __init__(self, num_classes, n_frames, n_shift, device):
    super(CNN_RNN, self).__init__()
    self.device = device
    self.n_frames = n_frames
    self.n_shift = n_shift
    self.conv1 = nn.Conv2d(1, 16, kernel_size=9, stride=1, padding=2)
    self.bn1= nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 16, kernel_size=4, stride=1, padding=2)
    self.dropoutCNN = nn.Dropout2d(p=0.3)
    self.bn2= nn.BatchNorm2d(16)
    self.gru = nn.GRUCell(input_size=16*28*14, hidden_size=8*28*14)
    self.dropoutRNN = nn.Dropout(p=0.5)
    self.fc2 = nn.Linear(8*28*14,num_classes)
  
  def forward(self, x):
    locuciones = x
    salida_lineal = []
    for locucion in locuciones:
      ventanas = self.calcular_ventanas_espectrales(locucion)
      cnn = []
      for ventana in ventanas:
        ventana = ventana.unsqueeze(0)
        ventana = ventana.unsqueeze(0)
        y = F.max_pool2d((F.leaky_relu(self.conv1(ventana))), kernel_size=3, stride=3, padding=0)
        y = F.max_pool2d(F.leaky_relu(self.conv2(y)), kernel_size=3, stride=3, padding=0)
        y = y.squeeze(0)
        cnn.append(y)
      y = torch.stack(cnn)
      y = y.flatten(start_dim=1)
      hx = torch.randn(1, 8*28*14).to(self.device)
      for i in range(y.shape[0]):
        hx = self.gru(y[i].unsqueeze(0), hx)
      y = self.fc2(hx)
      salida_lineal.append(y)
    samples = torch.stack(salida_lineal).squeeze(1)
    return samples

  def calcular_ventanas_espectrales(self,locucion):
    overlap = self.n_frames - self.n_shift
    locucion = locucion[1:,:].to(self.device)
    start = 0
    end = self.n_frames
    list_spectral_windows = []
    while end < locucion.shape[1] :
      list_spectral_windows.append(locucion[:,start:end])
      start = end - overlap
      end =  start + self.n_frames
    return list_spectral_windows