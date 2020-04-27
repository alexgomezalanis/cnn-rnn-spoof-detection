from torch.utils.data import DataLoader #A subpackage that contains utility classes like data sets and data loaders that make data preprocessing easier
from dataset import CNN_RNN_Dataset
import os
import sys
from utils.collate import collate
from model import CNN_RNN
import torch
import torch.nn as nn
import torch.optim as optim
from train import train
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
import matplotlib.pyplot as plt
from train import generate_confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def get_new_classes(all_labels,all_preds,device):

  mapping = {
  0: [0],
  1: [1,2],
  2: [2,1],
  3: [3],
  4: [4],
  5: [5],
  6: [6,7],
  7: [7,6],
  8: [8],
  9: [9],
  10: [10],
  11: [11] }

  newClasses = { 0:0, 1:1, 2:1, 3:2, 4:3,
                5:4,6:5,7:5,8:6,
                9:7,10:8,11:9 }

  # xx = { 'limpio':0,
  #       'clippingBajo':1, 'clippingMedio':2, 'clippingAlto':3,
  #       'reverberacionBaja':4, 'reverberacionMedia':5, 'reverberacionAlta':6,
  #       'SNRalto':7,'SNRmedio':8,'SNRbajo':9 }

  # LIST_CLASSES = ('limpio','clipping_09_5_percent','clipping_5_10_percent','clipping_20_40_percent','clipping_40_70_percent',
  #               'reverberacion_Lowhight','reverberacion_MediumLow','reverberacion_MediumHight','reverberacion_Hight',
  #               'noise_0_5db_SNR','noise_5_10db_SNR','noise_10_20db_SNR')
  new_labels = []
  new_preds = []
  correct = 0
  size = all_labels.size()[0]
  size2 = all_preds.size()[0]
  for i, label in enumerate(all_labels):
    label = label.item()
    pred = all_preds[i].item()
    if pred in mapping[label]: 
      correct += 1
    new_labels.append(newClasses[label])
    new_preds.append(newClasses[pred])
  #calculamos el nuevo accuracy
  accuracy = correct/size
  return accuracy, torch.tensor(new_labels).to(device),torch.tensor(new_preds).to(device)




labels = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11])
pred = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,11])

a,n_labels,n_preds = get_new_classes(labels,pred,torch.device('cpu'))

print('acurracy:',a)
print('labels:',n_labels)
print('preds:',n_preds)