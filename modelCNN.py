import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class CNN(nn.Module):
  def __init__(self, num_classes, n_frames, n_shift, device):
    super(CNN, self).__init__()
    self.device = device
    self.n_frames = n_frames
    self.n_shift = n_shift
    self.conv1 = nn.Conv2d(1, 32, kernel_size=9, stride=1, padding=2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2)
    #self.fc1 = nn.Linear(64*28*14,4*28*14)
    self.fc2 = nn.Linear(64*28*14,num_classes)
  
  def forward(self, x):
    locuciones = x
    salida_lineal = []
    for locucion in locuciones:
      ventanas = self.calcular_ventanas_espectrales(locucion)
      salida_por_ventana = []
      for ventana in ventanas:
        ventana = ventana.unsqueeze(0)
        ventana = ventana.unsqueeze(0)
        y = F.max_pool2d((F.leaky_relu(self.conv1(ventana))), kernel_size=3, stride=3, padding=0)
        y = F.max_pool2d(F.leaky_relu(self.conv2(y)), kernel_size=3, stride=3, padding=0)
        y = y.squeeze(0)
        y = y.flatten(start_dim=0)
        #y = self.fc1(F.leaky_relu(y))
        y = self.fc2(y)
        salida_por_ventana.append(y)
      promedio = self.promedio(salida_por_ventana)  
      salida_lineal.append(promedio)
    samples = torch.stack(salida_lineal)
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


  def promedio(self,salida_ventanas):
    y = torch.stack(salida_ventanas)
    return  y.sum(0)/y.shape[0]

