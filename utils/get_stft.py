import numpy as np
import soundfile as sf
import statistics as st
import math
from scipy import signal
import matplotlib.pyplot as plt


def plot_stft(t,f,file_path, Sxx,vmin=0):

    plt.pcolormesh(t, f, np.abs(Sxx), vmin=0)
    plt.title(file_path)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()



def get_stft(file_path, n_filts, n_frames, nperseg, noverlap, nfft):
  sig, fs = sf.read(file_path, dtype='float32')
  f, t, Sxx =  signal.stft(sig, fs, window='hamming', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
  Sxx = np.abs(Sxx)
  #aplicamos normalizacion a la entrada para controlar el rango de amplitudes variantes
  Sxx = (Sxx - np.mean(Sxx))/np.std(Sxx)
  features = np.where(Sxx > 1e-10, np.log10(Sxx), -10)
  return features