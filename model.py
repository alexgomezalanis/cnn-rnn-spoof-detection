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
    self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=2)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2)
    self.gru = nn.GRUCell(input_size=128*28*14, hidden_size=128*28*14) #input size = 1920
    self.fc2 = nn.Linear(128*28*14, num_classes) #input size = 1920 
  
  def forward(self, x, seq_length):
    #deep_features = []
    #for signal in signals:
    #  for window_context in 
    #    x = signal[:, ]
    # x = F.max_pool2d(F.sigmoid(self.conv1(x)), kernel_size=3, stride=3, padding=0)
    # x = F.max_pool2d(F.sigmoid(self.conv2(x)), kernel_size=3, stride=3, padding=0)
    # x = x.view(-1, 1, self.num_flat_features(x)) #reshape del tensor
    # hx = torch.randn(1, 1920).to(self.device)
    # for i in range(seq_length):
    #   hx = self.gru(x[i], hx)
    # x = self.fc2(hx)
    # return x
    locuciones = x[0]
    for locucion in locuciones:
      ventanas = self.calcular_ventanas_espectrales(locucion)
      cnn = [] 
      for ventana in ventanas:
        ventana = ventana.unsqueeze(0)
        ventana = ventana.unsqueeze(0)
        x = F.max_pool2d(F.sigmoid(self.conv1(ventana)), kernel_size=3, stride=3, padding=0)
        x = F.max_pool2d(F.sigmoid(self.conv2(x)), kernel_size=3, stride=3, padding=0)
        x = x.squeeze(0)
        cnn.append(x)
      salida_cnn = torch.stack(cnn)
      salida_cnn = salida_cnn.flatten()
      x = self.gru(salida_cnn)
      x = self.fc2(x)
      print(salida_cnn.shape)
    
  def calcular_ventanas_espectrales(self,locucion):
    overlap = self.n_frames - self.n_shift  #96: 128-32
    locucion = locucion[1:,:] #eliminamos la frecuencia O 257 --> 256
    start = 0
    end = self.n_frames
    list_spectral_windows = []
    list_indices = []
    while end < locucion.shape[1] :
      list_indices.append(start)
      list_indices.append(end)
      list_spectral_windows.append(locucion[:,start:end])
      start = end - overlap
      end =  start + self.n_frames
    return list_spectral_windows




  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features