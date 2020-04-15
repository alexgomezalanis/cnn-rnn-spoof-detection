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
    self.conv1 = nn.Conv2d(1, 8, kernel_size=9, stride=1, padding=2)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=2)
    self.gru = nn.GRUCell(input_size=16*28*14, hidden_size=16*28*14)
    self.fc2 = nn.Linear(16*28*14,num_classes)
  
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
        # print('procesado de una ventana espectral: ', ventana.shape)
        y = F.max_pool2d(F.leaky_relu(self.conv1(ventana)), kernel_size=3, stride=3, padding=0)
        y = F.max_pool2d(F.leaky_relu(self.conv2(y)), kernel_size=3, stride=3, padding=0)
        y = y.squeeze(0)
        # print('Salida de las CNN: ', y.shape)
        cnn.append(y)
      y = torch.stack(cnn)
      y = y.flatten(start_dim=1)
      # print('Entrada a la RNN: ', y.shape)
      hx = torch.randn(1, 16*28*14).to(self.device)
      for i in range(y.shape[0]):
        hx = self.gru(y[i].unsqueeze(0), hx)
      y = self.fc2(hx)
      salida_lineal.append(y)
    samples = torch.stack(salida_lineal).squeeze(1)
    # print('Salida final: ', samples.shape)
    return (samples, labels)

  def calcular_ventanas_espectrales(self,locucion):
    # print('--------CALCULO VENTANAS ESPECTRALES-------')
    # print('-------------------------------------')
    overlap = self.n_frames - self.n_shift  #96: 128-32
    locucion = locucion[1:,:].to(self.device) #eliminamos la frecuencia O 257 --> 256
    start = 0
    end = self.n_frames
    list_spectral_windows = []
    while end < locucion.shape[1] :
      list_spectral_windows.append(locucion[:,start:end])
      start = end - overlap
      end =  start + self.n_frames
    return list_spectral_windows