from torch.utils.data import DataLoader #A subpackage that contains utility classes like data sets and data loaders that make data preprocessing easier
from dataset import CNN_RNN_Dataset
import os
from utils.collate import collate
from model import CNN_RNN
import torch



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
n_frames=num_frames,
window=window_length,
shift=frame_shift,
dataset='training',
num_classes=num_classes)

muestra = next(iter(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
     num_workers=0, collate_fn=collate)


muestras = next(iter(train_loader))

print(type(muestras))

model = CNN_RNN(num_classes,n_frames,n_shift,device)


model(muestras,3)