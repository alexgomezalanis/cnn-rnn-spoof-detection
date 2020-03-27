import numpy as np
import soundfile as sf
import math
from scipy import signal
import matplotlib.pyplot as plt


def plot_stft(t,f,Sxx,vmin=0):

    plt.pcolormesh(t, f, np.abs(Sxx), vmin=0)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()



def get_stft(file_path, n_filts, n_frames, nperseg, noverlap, nfft):
  print('n_frames:',n_frames,'nperseg:',nperseg,'noverlap',noverlap,'nfft',nfft)
  #blackman
  sig, fs = sf.read(file_path, dtype='float32')
  print(len(sig))
  f, t, Sxx =  signal.stft(sig, fs, window='hamming', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
  Sxx = np.abs(Sxx)
  print('Sxx shape',len(Sxx))
  features = np.where(Sxx > 1e-10, np.log10(Sxx), -10)
  print(features)
  print(features.shape)
  data = np.reshape(features, (1, -1, n_filts)) # channel, frames, filters
  return data