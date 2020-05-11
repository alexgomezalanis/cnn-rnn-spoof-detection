import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class DNN(nn.Module):
  def __init__(self, num_classes, n_frames, n_shift, device):
    super(DNN, self).__init__()
    self.device = device
    self.n_frames = n_frames
    self.n_shift = n_shift
    self.fc1 = nn.Linear(1,1024)
    self.fc2 = nn.Linear(1,1024)
    self.fc3 = nn.Linear(1024,1024)
    self.fc4 = nn.Linear(1024,num_classes)
  
  def forward(self, x):
    locuciones = x
    salida_lineal = []
    for locucion in locuciones:
      ventanas = self.calcular_ventanas_espectrales(locucion)
      salida_por_ventana = []
      for ventana in ventanas:
        ventana = ventana.flatten(start_dim=0)
        y = self.fc1(F.leaky_relu(ventana))
        print('salida fc1:',y.shape)
        y = self.fc2(F.leaky_relu(y))
        print('salida fc2:',y.shape)
        y = self.fc3(F.leaky_relu(y))
        print('salida fc3:',y.shape)
        y = self.fc4(F.leaky_relu(y))
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

