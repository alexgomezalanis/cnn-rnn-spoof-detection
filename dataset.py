from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.get_stft import get_stft

PA_CLASSES = {'-': 0, 'AA': 1, 'AB': 2, 'AC': 3, 'BA': 4, 'BB': 5, 'BC': 6, 'CA': 7, 'CB': 8, 'CC': 9}

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
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the wav files.
        n_
        window length (float): Window length in seconds
        frame shift (float): Frame shift in seconds
    """
    if dataframe is not None:
      self.wavfiles_frame = dataframe
    else:
      self.wavfiles_frame = pd.read_csv(csv_file, sep=' ')
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
    label = self.wavfiles_frame['label'][idx]
    nameFile = self.wavfiles_frame['wav'][idx]
    
    if self.is_evaluating_la:
      if label == 'bonafide':
        label = 'A00'
      else:
        label = self.wavfiles_frame['target'][idx]
      label = label[1:]
      target = int(label)
    else:
      if self.num_classes == 10:
        target = PA_CLASSES[self.wavfiles_frame['target'][idx]]
      else:
        target = 0 if label == 'bonafide' else 1

    db = 'la-challenge' if self.is_evaluating_la else 'pa-challenge'
    path_db = os.path.join(self.root_dir, db)
    if self.dataset != 'test':
      if self.is_evaluating_la:
        file_dir = 'S' + str(target)
      else:
        file_dir = 'genuine' if label == 'bonafide' else 'spoof'
      file_path = os.path.join(path_db, 'flac-files', self.dataset, file_dir, nameFile + '.flac')
    else:
      file_path = os.path.join(path_db, 'flac-files', self.dataset, 'ASVspoof2019_' + nameFile[:2] + '_eval_v1/flac', nameFile + '.flac')

    # STFT features
    stft = get_stft(file_path, self.n_filts, self.n_frames, self.nperseg, self.noverlap, self.nfft)
    sample = (stft, target, nameFile)

    return sample