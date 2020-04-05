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
    self.conv2 = nn.Conv2d(64, 5, kernel_size=4, stride=1, padding=2)
    self.gru = nn.GRUCell(input_size=5*28*14, hidden_size=5*28*14) #input size = 128*28*14
    self.fc2 = nn.Linear(5*28*14,num_classes) #input size = 1920 
  
  def forward(self, x):
    locuciones = x[0]
    labels = torch.stack(x[1])
    salida_lineal = []
    for locucion in locuciones:
      ventanas = self.calcular_ventanas_espectrales(locucion)
      cnn = []
      for ventana in ventanas:
        ventana = ventana.unsqueeze(0) #añadimos una dimensión para decir que el numero de canales es 1
        ventana = ventana.unsqueeze(0) #añadimos otra dimensión para decir que el tamaño del lote es 1
        y = F.max_pool2d(torch.sigmoid(self.conv1(ventana)), kernel_size=3, stride=3, padding=0)
        y = F.max_pool2d(torch.sigmoid(self.conv2(y)), kernel_size=3, stride=3, padding=0)
        y = y.squeeze(0)
        cnn.append(y)
      y = torch.stack(cnn) #juntamos las diferentes embedding en un solo tensor [nº embeddings,nº canales,lenght,widht]
      y = y.flatten(start_dim=1)
      hx = torch.randn(1, 5*28*14).to(self.device)
      for i in range(y.shape[0]):
        hx = self.gru(y[i].unsqueeze(0), hx)
      y = self.fc2(hx)
      salida_lineal.append(y)
    tensor_salida = torch.stack(salida_lineal).squeeze(1)
    samples = F.softmax(tensor_salida,dim=1)
    return (samples, labels)

  def calcular_ventanas_espectrales(self,locucion):
    overlap = self.n_frames - self.n_shift  #96: 128-32
    locucion = locucion[1:,:] #eliminamos la frecuencia O 257 --> 256
    start = 0
    end = self.n_frames
    list_spectral_windows = []
    while end < locucion.shape[1] :
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