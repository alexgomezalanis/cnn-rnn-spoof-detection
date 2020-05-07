from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
#Dataset : Una clase abstracta para representar un conjunto de datos
#DataLoader: Envuelve un conjunto de datos y proporciona acceso a los datos subyacentes.
from utils.get_stft import get_stft

CLASSES = {'limpio': 0, 'clipping_05_09_percent': 1,
 'clipping_09_5_percent': 1,'clipping_5_10_percent': 2,'clipping_10_20_percent':3,'clipping_20_40_percent': 3, 'clipping_40_70_percent': 4,'clipping_70_90_percent':4,
 'reverberacion_LowLow': 5,'reverberacion_Lowhight': 5, 'reverberacion_MediumLow': 6, 'reverberacion_MediumHight': 7, 'reverberacion_Hight': 8,'reverberacion_HightHight': 8,
 'noise_0_5db_SNR': 9,'noise_5_10db_SNR': 10,'noise_10_20db_SNR': 11 }

LIST_CLASSES = ('limpio','clipping_09_5_percent','clipping_5_10_percent','clipping_20_40_percent','clipping_40_70_percent',
                'reverberacion_Lowhight','reverberacion_MediumLow','reverberacion_MediumHight','reverberacion_Hight',
                'noise_0_5db_SNR','noise_5_10db_SNR','noise_10_20db_SNR')
Fs = 16000 # Hz

class CNN_RNN_Dataset(Dataset): #creamos una clase que hereda de Dataset 
  #Específicamente, hay dos métodos que deben implementarse:
    #-- __len__método que devuelve la longitud del conjunto de datos
    #-- __getitem__método que obtiene un elemento del conjunto de datos en una ubicación de índice específica dentro del conjunto de datos.
  def __init__(self,
    csv_file,
    root_dir,
    n_filts,
    n_frames,
    window,
    shift,
    dataset,
    num_classes,
    dataframe=None):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the wav files.
        n_filts (): Number of filters considered in each frame
        n_frames (): Number of consecutive frames of a context
        window length (float): Window length in seconds
        frame shift (float): Frame shift in seconds
        dataset (string) [training, development, test]: training, development or test
    """
    if dataframe is not None:
      self.wavfiles_frame = dataframe
    else:
      self.wavfiles_frame = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.n_filts = n_filts
    self.n_frames = n_frames
    self.nperseg = int(window * Fs)
    self.noverlap = int(self.nperseg - shift * Fs) #shif*Fs  #original --> int(self.nperseg - shift * Fs)
    self.nfft = int(n_filts * 2)
    self.dataset = dataset
    self.num_classes = num_classes
    self.classes = LIST_CLASSES

  def __len__(self):
    return len(self.wavfiles_frame)

  def __getitem__(self, idx):
    
    label = self.wavfiles_frame['label'][idx]
    nameFile = self.wavfiles_frame['wav'][idx]
    target = CLASSES[label]

    if self.dataset == 'train':
      file_dir = 'ConjuntoDatosEntrenamiento'
    elif self.dataset == 'development':
      file_dir = 'ConjuntoDatosValidacion'
    elif self.dataset == 'mezcla':
      file_dir = 'ConjuntoDatosMezcla'
    elif self.dataset == 'audiosReales':
      file_dir = 'ConjuntoAudiosReales'
    elif self.dataset == 'ingles':
      file_dir = 'ConjuntoAudiosIngles'
    else:
      file_dir = 'ConjuntoDatosTest'

    file_path = os.path.join(self.root_dir, file_dir, nameFile)
    stft = get_stft(file_path, self.n_filts, self.n_frames, self.nperseg, self.noverlap, self.nfft)
    sample = (stft, target, nameFile)

    return sample