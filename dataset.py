from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.get_stft import get_stft

PA_CLASSES = {'-': 0, 'AA': 1, 'AB': 2, 'AC': 3, 'BA': 4, 'BB': 5, 'BC': 6, 'CA': 7, 'CB': 8, 'CC': 9}
PA_CLASSES_TFM = {'limpio': 0,
 'clipping_5_10_percent': 1, 'clipping_07_5_percent': 2, 'clipping_10_20_percent': 3, 'clipping_20_40_percent': 4, 'clipping_40_70_percent': 5,
 'reverberacion_Lowhight': 6, 'reverberacion_MediumLow': 7, 'reverberacion_MediumHight': 8, 'reverberacion_Hight': 9,
 'noise_0_5db_SNR': 10,'noise_5_10db_SNR': 11,'noise_10_20db_SNR': 12 }
Fs = 16000 # Hz

class CNN_RNN_Dataset(Dataset):
  def __init__(self,
    csv_file,
    root_dir,
    n_filts,
    n_frames,
    window,
    shift,
    dataset,
    is_evaluating_la,
    num_classes,
    dataframe=None):

    print(dataset)
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
    self.nperseg = window * Fs
    self.noverlap = self.nperseg - shift * Fs
    self.nfft = int(n_filts * 2)
    self.dataset = dataset
    self.is_evaluating_la = is_evaluating_la
    self.num_classes = num_classes

  def __len__(self):
    return len(self.wavfiles_frame)

  def __getitem__(self, idx):
    ## hay que leer el audio (wav) y hacerle la STFT:
      ## 1º sacar el label y nameWav asociado a un idx del .csv
        ##1.1 Conseguir el target asociado a la label
      ## 2º leer el archivo del directorio correspondiente
      ## 3º hacerle la STFT
    #Se devuelve STFT,TARGET y nombre del fichero
    label = self.wavfiles_frame['label'][idx]
    nameFile = self.wavfiles_frame['wav'][idx]
    target = PA_CLASSES_TFM[label]

    if self.dataset == 'train':
      file_dir = 'ConjuntoDatosEntrenamiento'
    elif self.dataset == 'development':
      file_dir = 'ConjuntoDatosValidacion'
    else:
      file_dir = 'ConjuntoDatosEntrenamiento'

    file_path = os.path.join(self.root_dir, file_dir, nameFile)


    # if self.is_evaluating_la:
    #   if label == 'bonafide':
    #     label = 'A00'
    #   else:
    #     label = self.wavfiles_frame['target'][idx]
    #   label = label[1:]
    #   target = int(label)
    # else:
    #   if self.num_classes == 10:
    #     target = PA_CLASSES[self.wavfiles_frame['target'][idx]]
    #   else:
    #     target = 0 if label == 'bonafide' else 1

    # db = 'la-challenge' if self.is_evaluating_la else 'pa-challenge'
    # path_db = os.path.join(self.root_dir, db)
    # if self.dataset != 'test':
    #   if self.is_evaluating_la:
    #     file_dir = 'S' + str(target)
    #   else:
    #     file_dir = 'genuine' if label == 'bonafide' else 'spoof'
    #   file_path = os.path.join(path_db, 'flac-files', self.dataset, file_dir, nameFile + '.flac')
    # else: #if dataset == 'test'
    #   file_path = os.path.join(path_db, 'flac-files', self.dataset, 'ASVspoof2019_' + nameFile[:2] + '_eval_v1/flac', nameFile + '.flac')

    # STFT features
    stft = get_stft(file_path, self.n_filts, self.n_frames, self.nperseg, self.noverlap, self.nfft)
    sample = (stft, target, nameFile)

    return sample