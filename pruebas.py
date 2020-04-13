from torch.utils.data import DataLoader #A subpackage that contains utility classes like data sets and data loaders that make data preprocessing easier
from dataset import CNN_RNN_Dataset
import os
import sys
from utils.collate import collate
from model import CNN_RNN
import torch
import torch.nn as nn
import torch.optim as optim



train_protocol = 'ConjuntoDatosEntrenamiento.csv'
dev_protocol = 'ConjuntoDatosValidacion.csv'
root_dir = './database'

print('Programa de pruebas:')
n_frames = 128
n_shift = 32
num_filts = 256
num_frames = 32
window_length = 0.025
frame_shift = 0.01
num_classes = 13
device = torch.device("cpu")

train_dataset = CNN_RNN_Dataset(
csv_file='./protocols/' + train_protocol,
root_dir=root_dir,
n_filts=num_filts,
n_frames=num_frames, #128
window=window_length,
shift=frame_shift, #32
dataset='training',
num_classes=num_classes)

model = CNN_RNN(num_classes,n_frames,n_shift,device)

muestra = next(iter(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True,
     num_workers=0, collate_fn=collate)

muestras = next(iter(train_loader))

model(muestras)




